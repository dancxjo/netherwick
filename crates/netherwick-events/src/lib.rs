use std::collections::VecDeque;

use anyhow::Result;
use netherwick_actions::{ActionPrimitive, ApproachTarget, InspectTarget, ReignInput, TurnDir};
use netherwick_autonomic::{SafetyDecision, SafetyReason};
use netherwick_core::{Provenance, TimeMs};
use netherwick_experience::{
    Experience, ExperienceLatent, FuturePrediction, Impression, Sensation,
};
use netherwick_llm::LlmTeaching;
use netherwick_memory::RecallBundle;
use netherwick_now::{LlmSense, MemorySense, Now};
use serde::{Deserialize, Serialize};
use serde_json::json;
use uuid::Uuid;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DriveName {
    BatteryHunger,
    DangerAvoidance,
    Curiosity,
    SocialInterest,
    Fatigue,
    UncertaintyPressure,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventKind {
    Tick,
    BodyState,
    Bump,
    Cliff,
    WheelDrop,
    BatteryLow,
    ChargingStarted,
    ChargingStopped,
    EyeFrame,
    EarFrame,
    SoundHeard,
    SpeechHeard,
    FaceDetected,
    FaceRecognized,
    VoiceRecognized,
    NearWall,
    MemoryRecalled,
    PredictionMade,
    SurpriseHigh,
    LlmCommanded,
    LlmCritiqued,
    SafetyVetoed,
    SimDeadBatteryReset,
    SimStuckReset,
    ReignCommanded,
    ReignExpired,
    ReignCleared,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum EventPayload {
    #[default]
    None,
    BatteryLow {
        battery_level: f32,
    },
    ChargingStarted {
        battery_level: f32,
    },
    ChargingStopped,
    Bump {
        side: String,
    },
    SoundHeard {
        sources: usize,
    },
    FaceDetected {
        face_embeddings: usize,
    },
    Cliff,
    WheelDrop,
    SpeechHeard {
        transcript: String,
    },
    FaceRecognized {
        face_embeddings: usize,
        face_familiarity: f32,
    },
    MemoryRecalled {
        hits: usize,
    },
    PredictionMade {
        expected_events: Vec<String>,
        uncertainty: f32,
    },
    SurpriseHigh {
        total: f32,
    },
    LlmCommanded {
        summary: String,
    },
    LlmCritiqued {
        critique: String,
        confidence: f32,
    },
    SafetyVetoed {
        desired_action: ActionPrimitive,
        reason: String,
    },
    SimDeadBatteryReset {
        battery_level: f32,
    },
    SimStuckReset {
        corner_trap: bool,
        stuck_ticks: usize,
    },
    ReignCommanded {
        input: ReignInput,
    },
    ReignExpired {
        input: ReignInput,
    },
    ReignCleared,
    NearWall {
        distance_norm: f32,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Event {
    pub id: Uuid,
    pub t_ms: TimeMs,
    pub kind: EventKind,
    pub summary: Option<String>,
    pub provenance: Provenance,
    pub payload: EventPayload,
}

impl Event {
    pub fn new(t_ms: TimeMs, kind: EventKind) -> Self {
        Self {
            id: Uuid::new_v4(),
            t_ms,
            kind,
            summary: None,
            provenance: Provenance::direct().with_stage("event"),
            payload: EventPayload::None,
        }
    }

    pub fn with_summary(mut self, summary: impl Into<String>) -> Self {
        self.summary = Some(summary.into());
        self
    }

    pub fn with_payload(mut self, payload: EventPayload) -> Self {
        self.payload = payload;
        self
    }

    pub fn with_provenance(mut self, provenance: Provenance) -> Self {
        self.provenance = provenance;
        self
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Response {
    AddSensation(Sensation),
    AddImpression(Impression),
    AddExperience(Experience),
    AddMemoryNote(String),
    ProposeAction(ActionPrimitive),
    SetDrive { name: DriveName, value: f32 },
    SetMemorySense(MemorySense),
    Teach(LlmTeaching),
    Emit(Event),
}

impl Response {
    pub fn as_action(&self) -> Option<&ActionPrimitive> {
        match self {
            Self::ProposeAction(action) => Some(action),
            _ => None,
        }
    }
}

pub struct EventContext<'a> {
    pub now: &'a Now,
    pub latent: Option<&'a ExperienceLatent>,
    pub recall: Option<&'a RecallBundle>,
    pub predicted_futures: &'a [FuturePrediction],
    pub llm: Option<&'a LlmSense>,
    pub safety: Option<&'a SafetyDecision>,
}

pub trait EventResponder: Send {
    fn id(&self) -> &'static str;
    fn matches(&self, event: &Event) -> bool;
    fn respond(&mut self, ctx: &EventContext, event: &Event) -> Result<Vec<Response>>;
}

#[derive(Default)]
pub struct EventBus {
    responders: Vec<Box<dyn EventResponder>>,
}

impl EventBus {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn on<R>(&mut self, responder: R)
    where
        R: EventResponder + 'static,
    {
        self.responders.push(Box::new(responder));
    }

    pub fn dispatch(&mut self, ctx: &EventContext, event: &Event) -> Result<Vec<Response>> {
        let mut out = Vec::new();
        for responder in &mut self.responders {
            if responder.matches(event) {
                out.extend(responder.respond(ctx, event)?);
            }
        }
        Ok(out)
    }

    pub fn dispatch_all(
        &mut self,
        ctx: &EventContext,
        events: impl IntoIterator<Item = Event>,
    ) -> Result<DispatchOutput> {
        let mut queue = events.into_iter().collect::<VecDeque<_>>();
        let mut seen = Vec::new();
        let mut responses = Vec::new();

        while let Some(event) = queue.pop_front() {
            let emitted = self.dispatch(ctx, &event)?;
            for response in &emitted {
                if let Response::Emit(event) = response {
                    queue.push_back(event.clone());
                }
            }
            seen.push(event);
            responses.extend(emitted);
        }

        Ok(DispatchOutput {
            events: seen,
            responses,
        })
    }
}

pub struct DispatchOutput {
    pub events: Vec<Event>,
    pub responses: Vec<Response>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EventExtractorConfig {
    pub low_battery_threshold: f32,
    pub face_familiarity_threshold: f32,
    pub surprise_threshold: f32,
    pub near_wall_threshold: f32,
}

impl Default for EventExtractorConfig {
    fn default() -> Self {
        Self {
            low_battery_threshold: 0.20,
            face_familiarity_threshold: 0.70,
            surprise_threshold: 0.75,
            near_wall_threshold: 0.18,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct EventExtractor {
    pub config: EventExtractorConfig,
    last_now: Option<Now>,
    last_reign_id: Option<Uuid>,
    last_reign_input: Option<ReignInput>,
    last_reign_clear_sequence: u64,
}

impl EventExtractor {
    pub fn events_from_now(&mut self, now: &Now, recall: Option<&RecallBundle>) -> Vec<Event> {
        let mut events = Vec::new();
        let previous = self.last_now.as_ref();

        if now.reign.clear_sequence > self.last_reign_clear_sequence {
            events.push(
                Event::new(now.t_ms, EventKind::ReignCleared)
                    .with_summary("Human reign commands cleared.")
                    .with_payload(EventPayload::ReignCleared)
                    .with_provenance(Provenance::direct().with_stage("reign")),
            );
            self.last_reign_clear_sequence = now.reign.clear_sequence;
            self.last_reign_id = None;
            self.last_reign_input = None;
        }

        if now.reign.active {
            if let Some(input) = &now.reign.latest {
                if Some(input.id) != self.last_reign_id {
                    events.push(
                        Event::new(now.t_ms, EventKind::ReignCommanded)
                            .with_summary(format!("Human reign command: {:?}.", input.command))
                            .with_payload(EventPayload::ReignCommanded {
                                input: input.clone(),
                            })
                            .with_provenance(Provenance::direct().with_stage("reign")),
                    );
                    self.last_reign_id = Some(input.id);
                    self.last_reign_input = Some(input.clone());
                }
            }
        } else if let Some(input) = self.last_reign_input.take() {
            if now.reign.clear_sequence <= self.last_reign_clear_sequence {
                events.push(
                    Event::new(now.t_ms, EventKind::ReignExpired)
                        .with_summary("Human reign command expired.")
                        .with_payload(EventPayload::ReignExpired { input })
                        .with_provenance(Provenance::direct().with_stage("reign")),
                );
            }
            self.last_reign_id = None;
        }

        if now.body.battery_level <= self.config.low_battery_threshold {
            events.push(
                Event::new(now.t_ms, EventKind::BatteryLow)
                    .with_summary(format!("Battery low at {:.2}.", now.body.battery_level))
                    .with_payload(EventPayload::BatteryLow {
                        battery_level: now.body.battery_level,
                    }),
            );
        }

        if now.body.battery_level <= f32::EPSILON && !now.body.charging {
            events.push(
                Event::new(now.t_ms, EventKind::SimDeadBatteryReset)
                    .with_summary(
                        "Virtual battery is dead while not charging; simulation reset needed.",
                    )
                    .with_payload(EventPayload::SimDeadBatteryReset {
                        battery_level: now.body.battery_level,
                    })
                    .with_provenance(Provenance::direct().with_stage("sim")),
            );
        }

        if let Some(stuck_values) = now
            .extensions
            .get("sim.stuck")
            .and_then(|value| value.get("values"))
            .and_then(|value| value.as_array())
        {
            let reset_due = stuck_values
                .get(9)
                .and_then(|value| value.as_f64())
                .unwrap_or(0.0)
                > 0.0;
            if reset_due {
                let corner_trap = stuck_values
                    .get(1)
                    .and_then(|value| value.as_f64())
                    .unwrap_or(0.0)
                    > 0.0;
                let stuck_ticks = stuck_values
                    .get(2)
                    .and_then(|value| value.as_f64())
                    .unwrap_or(0.0)
                    .max(0.0) as usize;
                events.push(
                    Event::new(now.t_ms, EventKind::SimStuckReset)
                        .with_summary(
                            "Virtual body is still stuck after recovery; simulation reset needed.",
                        )
                        .with_payload(EventPayload::SimStuckReset {
                            corner_trap,
                            stuck_ticks,
                        })
                        .with_provenance(Provenance::direct().with_stage("sim")),
                );
            }
        }

        if now.body.charging && previous.map(|prior| !prior.body.charging).unwrap_or(false) {
            events.push(
                Event::new(now.t_ms, EventKind::ChargingStarted)
                    .with_summary("Charging started.")
                    .with_payload(EventPayload::ChargingStarted {
                        battery_level: now.body.battery_level,
                    }),
            );
        }

        if !now.body.charging && previous.map(|prior| prior.body.charging).unwrap_or(false) {
            events.push(
                Event::new(now.t_ms, EventKind::ChargingStopped)
                    .with_summary("Charging stopped.")
                    .with_payload(EventPayload::ChargingStopped),
            );
        }

        if now.body.flags.bump_left || now.body.flags.bump_right {
            let side = bump_side(now).to_string();
            events.push(
                Event::new(now.t_ms, EventKind::Bump)
                    .with_summary(format!("Bump on the {side} side."))
                    .with_payload(EventPayload::Bump { side }),
            );
        }

        if now
            .range
            .beams
            .iter()
            .copied()
            .any(|beam| beam <= self.config.near_wall_threshold)
        {
            let distance_norm = now.range.beams.iter().copied().fold(1.0_f32, f32::min);
            events.push(
                Event::new(now.t_ms, EventKind::NearWall)
                    .with_summary(format!(
                        "Near wall at normalized range {:.2}.",
                        distance_norm
                    ))
                    .with_payload(EventPayload::NearWall { distance_norm }),
            );
        }

        if now.body.flags.cliff_left || now.body.flags.cliff_right {
            events.push(
                Event::new(now.t_ms, EventKind::Cliff)
                    .with_summary("Cliff detected.")
                    .with_payload(EventPayload::Cliff),
            );
        }

        if now.body.flags.wheel_drop {
            events.push(
                Event::new(now.t_ms, EventKind::WheelDrop)
                    .with_summary("Wheel drop detected.")
                    .with_payload(EventPayload::WheelDrop),
            );
        }

        if let Some(transcript) = &now.ear.transcript {
            let transcript = transcript.trim();
            if !transcript.is_empty() {
                events.push(
                    Event::new(now.t_ms, EventKind::SpeechHeard)
                        .with_summary(format!("Speech heard: {transcript}"))
                        .with_payload(EventPayload::SpeechHeard {
                            transcript: transcript.to_string(),
                        }),
                );
            }
        }

        if !now.ear.features.is_empty() {
            events.push(
                Event::new(now.t_ms, EventKind::SoundHeard)
                    .with_summary("Sound heard.")
                    .with_payload(EventPayload::SoundHeard {
                        sources: now.ear.features.len(),
                    }),
            );
        }

        let face_familiarity = recall
            .map(|bundle| bundle.sense.face_familiarity)
            .unwrap_or(now.memory.face_familiarity);
        if !now.face.embeddings.is_empty() {
            events.push(
                Event::new(now.t_ms, EventKind::FaceDetected)
                    .with_summary("Face detected.")
                    .with_payload(EventPayload::FaceDetected {
                        face_embeddings: now.face.embeddings.len(),
                    }),
            );
        }
        if !now.face.embeddings.is_empty()
            && face_familiarity >= self.config.face_familiarity_threshold
        {
            events.push(
                Event::new(now.t_ms, EventKind::FaceRecognized)
                    .with_summary("Recognized a familiar face.")
                    .with_payload(EventPayload::FaceRecognized {
                        face_embeddings: now.face.embeddings.len(),
                        face_familiarity,
                    }),
            );
        }

        if let Some(recall) = recall {
            if !recall.hits.is_empty() {
                events.push(
                    Event::new(now.t_ms, EventKind::MemoryRecalled)
                        .with_summary(format!("Recalled {} memory hits.", recall.hits.len()))
                        .with_payload(EventPayload::MemoryRecalled {
                            hits: recall.hits.len(),
                        }),
                );
            }
        }

        if !now.predictions.expected_events.is_empty() || now.predictions.uncertainty > 0.0 {
            events.push(
                Event::new(now.t_ms, EventKind::PredictionMade)
                    .with_summary("Predictions are active.")
                    .with_payload(EventPayload::PredictionMade {
                        expected_events: now.predictions.expected_events.clone(),
                        uncertainty: now.predictions.uncertainty,
                    }),
            );
        }

        if now.surprise.total >= self.config.surprise_threshold {
            events.push(
                Event::new(now.t_ms, EventKind::SurpriseHigh)
                    .with_summary(format!("Surprise high at {:.2}.", now.surprise.total))
                    .with_payload(EventPayload::SurpriseHigh {
                        total: now.surprise.total,
                    }),
            );
        }

        if let Some(summary) = &now.llm.command_summary {
            if !summary.trim().is_empty() {
                events.push(
                    Event::new(now.t_ms, EventKind::LlmCommanded)
                        .with_summary(summary.clone())
                        .with_payload(EventPayload::LlmCommanded {
                            summary: summary.clone(),
                        }),
                );
            }
        }

        if let Some(critique) = &now.llm.critique {
            if !critique.trim().is_empty() {
                events.push(
                    Event::new(now.t_ms, EventKind::LlmCritiqued)
                        .with_summary(critique.clone())
                        .with_payload(EventPayload::LlmCritiqued {
                            critique: critique.clone(),
                            confidence: now.llm.confidence,
                        }),
                );
            }
        }

        self.last_now = Some(now.clone());
        events
    }

    pub fn events_from_safety(
        &self,
        now: &Now,
        desired: &ActionPrimitive,
        safety: &SafetyDecision,
    ) -> Vec<Event> {
        if !safety.vetoed {
            return Vec::new();
        }
        vec![safety_veto_event(now, desired, safety.reason.clone())]
    }
}

fn bump_side(now: &Now) -> &'static str {
    match (now.body.flags.bump_left, now.body.flags.bump_right) {
        (true, false) => "left",
        (false, true) => "right",
        _ => "both",
    }
}

fn event_sensation(
    event: &Event,
    kind: &str,
    source: &str,
    summary: impl Into<String>,
    payload: serde_json::Value,
) -> Sensation {
    Sensation::new(kind, source, event.t_ms, event.t_ms, payload)
        .with_summary(summary)
        .with_provenance(event.provenance.clone().with_stage("responder"))
}

pub mod responders {
    use super::*;

    #[derive(Default)]
    pub struct BatteryLowResponder;

    impl EventResponder for BatteryLowResponder {
        fn id(&self) -> &'static str {
            "battery-low"
        }

        fn matches(&self, event: &Event) -> bool {
            event.kind == EventKind::BatteryLow
        }

        fn respond(&mut self, _ctx: &EventContext, event: &Event) -> Result<Vec<Response>> {
            Ok(vec![
                Response::ProposeAction(ActionPrimitive::Dock),
                Response::AddSensation(event_sensation(
                    event,
                    "drive.battery_hunger",
                    "body",
                    "I feel hungry for power.",
                    json!({ "kind": "battery_low" }),
                )),
                Response::AddMemoryNote("Battery low here.".to_string()),
                Response::SetDrive {
                    name: DriveName::BatteryHunger,
                    value: 1.0,
                },
            ])
        }
    }

    #[derive(Default)]
    pub struct ChargingStartedResponder;

    impl EventResponder for ChargingStartedResponder {
        fn id(&self) -> &'static str {
            "charging-started"
        }

        fn matches(&self, event: &Event) -> bool {
            event.kind == EventKind::ChargingStarted
        }

        fn respond(&mut self, _ctx: &EventContext, event: &Event) -> Result<Vec<Response>> {
            Ok(vec![
                Response::AddSensation(event_sensation(
                    event,
                    "body.charging",
                    "body",
                    "Charging feels sweet.",
                    json!({ "charging": true }),
                )),
                Response::AddMemoryNote("Charging tastes sweet here.".to_string()),
                Response::AddExperience(Experience::new(
                    "body.charging",
                    "Charging happened here and felt sweet.",
                    Vec::new(),
                    Vec::new(),
                    event.t_ms,
                    event.t_ms,
                )),
                Response::SetDrive {
                    name: DriveName::BatteryHunger,
                    value: 0.0,
                },
            ])
        }
    }

    #[derive(Default)]
    pub struct BumpResponder;

    impl EventResponder for BumpResponder {
        fn id(&self) -> &'static str {
            "bump"
        }

        fn matches(&self, event: &Event) -> bool {
            event.kind == EventKind::Bump
        }

        fn respond(&mut self, _ctx: &EventContext, event: &Event) -> Result<Vec<Response>> {
            let side = match &event.payload {
                EventPayload::Bump { side } => side.as_str(),
                _ => "unknown",
            };
            let direction = match side {
                "left" => TurnDir::Right,
                _ => TurnDir::Left,
            };
            Ok(vec![
                Response::AddSensation(event_sensation(
                    event,
                    "body.bump",
                    "body",
                    format!("I bumped my {side} side."),
                    json!({ "side": side }),
                )),
                Response::AddMemoryNote("Danger here.".to_string()),
                Response::SetDrive {
                    name: DriveName::DangerAvoidance,
                    value: 1.0,
                },
                Response::ProposeAction(ActionPrimitive::Turn {
                    direction,
                    intensity: 0.5,
                    duration_ms: 1_000,
                }),
            ])
        }
    }

    #[derive(Default)]
    pub struct MemoryRecalledResponder;

    impl EventResponder for MemoryRecalledResponder {
        fn id(&self) -> &'static str {
            "memory-recalled"
        }

        fn matches(&self, event: &Event) -> bool {
            event.kind == EventKind::MemoryRecalled
        }

        fn respond(&mut self, ctx: &EventContext, event: &Event) -> Result<Vec<Response>> {
            let Some(recall) = ctx.recall else {
                return Ok(Vec::new());
            };
            let hits = match event.payload {
                EventPayload::MemoryRecalled { hits } => hits,
                _ => recall.hits.len(),
            };
            let mut responses = recall
                .recollections
                .iter()
                .map(|recollection| Response::AddSensation(recollection.sensation.clone()))
                .collect::<Vec<_>>();
            responses.push(Response::SetMemorySense(recall.sense.clone()));
            responses.push(Response::AddSensation(event_sensation(
                event,
                "memory.recall",
                "memory",
                recall.first_person_summary.clone(),
                json!({ "hits": hits }),
            )));
            Ok(responses)
        }
    }

    #[derive(Default)]
    pub struct SurpriseHighResponder;

    impl EventResponder for SurpriseHighResponder {
        fn id(&self) -> &'static str {
            "surprise-high"
        }

        fn matches(&self, event: &Event) -> bool {
            event.kind == EventKind::SurpriseHigh
        }

        fn respond(&mut self, _ctx: &EventContext, event: &Event) -> Result<Vec<Response>> {
            Ok(vec![
                Response::AddSensation(event_sensation(
                    event,
                    "surprise.attend",
                    "prediction",
                    "This is surprising. I should attend and remember it.",
                    json!({ "kind": "surprise_high" }),
                )),
                Response::AddMemoryNote("This moment was surprisingly important.".to_string()),
                Response::SetDrive {
                    name: DriveName::Curiosity,
                    value: 0.8,
                },
                Response::ProposeAction(ActionPrimitive::Inspect {
                    target: InspectTarget::Novelty,
                }),
            ])
        }
    }

    #[derive(Default)]
    pub struct SafetyVetoResponder;

    impl EventResponder for SafetyVetoResponder {
        fn id(&self) -> &'static str {
            "safety-veto"
        }

        fn matches(&self, event: &Event) -> bool {
            event.kind == EventKind::SafetyVetoed
        }

        fn respond(&mut self, _ctx: &EventContext, event: &Event) -> Result<Vec<Response>> {
            let reason = match &event.payload {
                EventPayload::SafetyVetoed { reason, .. } => reason.as_str(),
                _ => "unknown reason",
            };
            Ok(vec![
                Response::AddExperience(Experience::new(
                    "safety.veto",
                    format!("My body refused the command because {reason}."),
                    Vec::new(),
                    Vec::new(),
                    event.t_ms,
                    event.t_ms,
                )),
                Response::AddMemoryNote(format!("Safety veto happened because {reason}.")),
                Response::Teach(LlmTeaching {
                    t_ms: event.t_ms,
                    summary: format!("Safety vetoed an action because {reason}."),
                    critique: Some("Body safety overrode the chosen action.".to_string()),
                    counterfactuals: Vec::new(),
                    memory_notes: vec![format!(
                        "Avoid repeating action after safety veto: {reason}"
                    )],
                    confidence: 0.9,
                }),
            ])
        }
    }

    #[derive(Default)]
    pub struct SimDeadBatteryResponder;

    impl EventResponder for SimDeadBatteryResponder {
        fn id(&self) -> &'static str {
            "sim-dead-battery"
        }

        fn matches(&self, event: &Event) -> bool {
            event.kind == EventKind::SimDeadBatteryReset
        }

        fn respond(&mut self, _ctx: &EventContext, event: &Event) -> Result<Vec<Response>> {
            Ok(vec![
                Response::AddMemoryNote(
                    "VirtualDeadBattery: battery hit 0 while not charging; simulation reset queued."
                        .to_string(),
                ),
                Response::Teach(LlmTeaching {
                    t_ms: event.t_ms,
                    summary:
                        "The virtual body ran out of battery away from charge, so the simulation reset."
                            .to_string(),
                    critique: Some(
                        "Dead battery away from the charger ends the run; seek charge earlier."
                            .to_string(),
                    ),
                    counterfactuals: Vec::new(),
                    memory_notes: vec![
                        "When battery is critical and not charging, prioritize docking immediately."
                            .to_string(),
                    ],
                    confidence: 0.95,
                }),
            ])
        }
    }

    #[derive(Default)]
    pub struct SimStuckResponder;

    impl EventResponder for SimStuckResponder {
        fn id(&self) -> &'static str {
            "sim-stuck"
        }

        fn matches(&self, event: &Event) -> bool {
            event.kind == EventKind::SimStuckReset
        }

        fn respond(&mut self, _ctx: &EventContext, event: &Event) -> Result<Vec<Response>> {
            let (corner_trap, stuck_ticks) = match &event.payload {
                EventPayload::SimStuckReset {
                    corner_trap,
                    stuck_ticks,
                } => (*corner_trap, *stuck_ticks),
                _ => (false, 0),
            };
            let class = if corner_trap { "corner trap" } else { "stuck" };
            Ok(vec![
                Response::AddMemoryNote(format!(
                    "VirtualStuck: {class} persisted after recovery and {stuck_ticks} low-displacement ticks; simulation reset queued."
                )),
                Response::Teach(LlmTeaching {
                    t_ms: event.t_ms,
                    summary: format!(
                        "The virtual body stayed in {class} after recovery, so the simulation reset."
                    ),
                    critique: Some(format!(
                        "Getting {class} after recovery ends the run; avoid repeated motion into blocked space."
                    )),
                    counterfactuals: Vec::new(),
                    memory_notes: vec![
                        "When motion produces no displacement near obstacles, back off and choose a new route."
                            .to_string(),
                    ],
                    confidence: 0.95,
                }),
            ])
        }
    }

    #[derive(Default)]
    pub struct ReignResponder;

    impl EventResponder for ReignResponder {
        fn id(&self) -> &'static str {
            "reign.default_responder"
        }

        fn matches(&self, event: &Event) -> bool {
            event.kind == EventKind::ReignCommanded
        }

        fn respond(&mut self, _ctx: &EventContext, event: &Event) -> Result<Vec<Response>> {
            let input = match &event.payload {
                EventPayload::ReignCommanded { input } => input,
                _ => return Ok(Vec::new()),
            };
            let Some(action) = input.command.to_action() else {
                return Ok(vec![Response::AddSensation(event_sensation(
                    event,
                    "reign.mode",
                    "human",
                    "I received a remote reign mode command.",
                    json!({ "input": input }),
                ))]);
            };
            Ok(vec![
                Response::AddSensation(event_sensation(
                    event,
                    "reign.command",
                    "human",
                    "I received a remote reign command.",
                    json!({ "input": input }),
                )),
                Response::ProposeAction(action),
            ])
        }
    }

    #[derive(Default)]
    pub struct FaceRecognizedResponder;

    impl EventResponder for FaceRecognizedResponder {
        fn id(&self) -> &'static str {
            "face-recognized"
        }

        fn matches(&self, event: &Event) -> bool {
            event.kind == EventKind::FaceRecognized
        }

        fn respond(&mut self, ctx: &EventContext, event: &Event) -> Result<Vec<Response>> {
            let name = ctx
                .recall
                .and_then(|bundle| bundle.hits.first())
                .map(|hit| hit.summary.clone())
                .filter(|summary| !summary.trim().is_empty())
                .unwrap_or_else(|| "someone familiar".to_string());
            let action = if ctx.now.drives.danger_avoidance >= 0.7 {
                ActionPrimitive::Inspect {
                    target: InspectTarget::Person,
                }
            } else if ctx.now.drives.social_interest >= 0.7 {
                ActionPrimitive::Speak {
                    text: "Hello again.".to_string(),
                }
            } else {
                ActionPrimitive::Approach {
                    target: ApproachTarget::Person,
                }
            };
            Ok(vec![
                Response::AddSensation(event_sensation(
                    event,
                    "face.recognized",
                    "vision",
                    format!("I recognize {name}."),
                    json!({ "name": name }),
                )),
                Response::ProposeAction(action),
            ])
        }
    }
}

pub fn default_event_bus() -> EventBus {
    let mut bus = EventBus::new();
    bus.on(responders::ReignResponder);
    bus.on(responders::BatteryLowResponder);
    bus.on(responders::ChargingStartedResponder);
    bus.on(responders::BumpResponder);
    bus.on(responders::MemoryRecalledResponder);
    bus.on(responders::SurpriseHighResponder);
    bus.on(responders::SafetyVetoResponder);
    bus.on(responders::SimDeadBatteryResponder);
    bus.on(responders::SimStuckResponder);
    bus.on(responders::FaceRecognizedResponder);
    bus
}

pub fn safety_veto_event(
    now: &Now,
    desired: &ActionPrimitive,
    reason: Option<SafetyReason>,
) -> Event {
    let reason = reason
        .map(|value| format!("{value:?}"))
        .unwrap_or_else(|| "Unknown".to_string());
    Event::new(now.t_ms, EventKind::SafetyVetoed)
        .with_summary(format!("Safety vetoed {:?}.", desired))
        .with_payload(EventPayload::SafetyVetoed {
            desired_action: desired.clone(),
            reason,
        })
        .with_provenance(Provenance::direct().with_stage("safety"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use netherwick_body::BodySense;
    use netherwick_memory::RecallBundle;

    #[test]
    fn extractor_emits_memory_recalled_when_hits_exist() {
        let mut extractor = EventExtractor::default();
        let now = Now::blank(7, BodySense::default());
        let recall = RecallBundle {
            hits: vec![netherwick_now::RecallHit {
                frame_id: None,
                score: 0.9,
                summary: "danger".to_string(),
                warning: None,
                graph_context: Vec::new(),
            }],
            ..RecallBundle::default()
        };

        let events = extractor.events_from_now(&now, Some(&recall));

        assert!(events
            .iter()
            .any(|event| event.kind == EventKind::MemoryRecalled));
    }

    #[test]
    fn extractor_emits_reign_commanded_once_then_expired() {
        let mut extractor = EventExtractor::default();
        let mut now = Now::blank(20, BodySense::default());
        let input = test_reign_input(20);
        now.reign.active = true;
        now.reign.latest = Some(input.clone());

        let first = extractor.events_from_now(&now, None);
        let second = extractor.events_from_now(&now, None);
        now.t_ms = 1_200;
        now.reign.active = false;
        now.reign.latest = None;
        let expired = extractor.events_from_now(&now, None);

        assert!(first
            .iter()
            .any(|event| event.kind == EventKind::ReignCommanded));
        assert!(!second
            .iter()
            .any(|event| event.kind == EventKind::ReignCommanded));
        assert!(expired.iter().any(|event| {
            matches!(
                &event.payload,
                EventPayload::ReignExpired { input: expired_input }
                    if expired_input.id == input.id
            )
        }));
    }

    #[test]
    fn extractor_emits_reign_cleared_without_expired() {
        let mut extractor = EventExtractor::default();
        let mut now = Now::blank(20, BodySense::default());
        now.reign.active = true;
        now.reign.latest = Some(test_reign_input(20));
        let _ = extractor.events_from_now(&now, None);

        now.t_ms = 30;
        now.reign.active = false;
        now.reign.latest = None;
        now.reign.clear_sequence = 1;
        let events = extractor.events_from_now(&now, None);

        assert!(events
            .iter()
            .any(|event| event.kind == EventKind::ReignCleared));
        assert!(!events
            .iter()
            .any(|event| event.kind == EventKind::ReignExpired));
    }

    #[test]
    fn battery_low_responder_proposes_dock() {
        let mut bus = EventBus::new();
        bus.on(responders::BatteryLowResponder);
        let mut body = BodySense::default();
        body.battery_level = 0.1;
        let now = Now::blank(5, body);
        let ctx = EventContext {
            now: &now,
            latent: None,
            recall: None,
            predicted_futures: &[],
            llm: None,
            safety: None,
        };
        let event = Event::new(5, EventKind::BatteryLow)
            .with_payload(EventPayload::BatteryLow { battery_level: 0.1 });

        let output = bus.dispatch(&ctx, &event).unwrap();

        assert!(output
            .iter()
            .any(|response| matches!(response, Response::ProposeAction(ActionPrimitive::Dock))));
    }

    #[test]
    fn extractor_emits_near_wall_face_and_sound_events() {
        let mut extractor = EventExtractor::default();
        let mut now = Now::blank(12, BodySense::default());
        now.range.beams = vec![0.12, 0.8];
        now.ear.features = vec![vec![0.7]];
        now.face.embeddings = vec![vec![0.1, 0.2, 0.3]];
        now.memory.face_familiarity = 0.9;

        let events = extractor.events_from_now(&now, None);

        assert!(events.iter().any(|event| event.kind == EventKind::NearWall));
        assert!(events
            .iter()
            .any(|event| event.kind == EventKind::SoundHeard));
        assert!(events
            .iter()
            .any(|event| event.kind == EventKind::FaceDetected));
        assert!(events
            .iter()
            .any(|event| event.kind == EventKind::FaceRecognized));
    }

    #[test]
    fn charging_started_responder_adds_sweetness_and_memory_note() {
        let mut bus = EventBus::new();
        bus.on(responders::ChargingStartedResponder);
        let mut body = BodySense::default();
        body.charging = true;
        let now = Now::blank(8, body);
        let ctx = EventContext {
            now: &now,
            latent: None,
            recall: None,
            predicted_futures: &[],
            llm: None,
            safety: None,
        };
        let event = Event::new(8, EventKind::ChargingStarted)
            .with_payload(EventPayload::ChargingStarted { battery_level: 0.8 });

        let output = bus.dispatch(&ctx, &event).unwrap();

        assert!(output.iter().any(|response| match response {
            Response::AddSensation(sensation) => sensation
                .summary
                .as_deref()
                .unwrap_or_default()
                .contains("sweet"),
            _ => false,
        }));
        assert!(output.iter().any(
            |response| matches!(response, Response::AddMemoryNote(note) if note.contains("sweet"))
        ));
    }

    #[test]
    fn bump_responder_marks_danger_and_escape() {
        let mut bus = EventBus::new();
        bus.on(responders::BumpResponder);
        let now = Now::blank(9, BodySense::default());
        let ctx = EventContext {
            now: &now,
            latent: None,
            recall: None,
            predicted_futures: &[],
            llm: None,
            safety: None,
        };
        let event = Event::new(9, EventKind::Bump).with_payload(EventPayload::Bump {
            side: "left".to_string(),
        });

        let output = bus.dispatch(&ctx, &event).unwrap();

        assert!(output.iter().any(|response| match response {
            Response::AddSensation(sensation) => sensation.kind == "body.bump",
            _ => false,
        }));
        assert!(output.iter().any(|response| matches!(
            response,
            Response::ProposeAction(ActionPrimitive::Turn {
                direction: TurnDir::Right,
                ..
            })
        )));
    }

    #[test]
    fn memory_recalled_responder_updates_memory_sense() {
        let mut bus = EventBus::new();
        bus.on(responders::MemoryRecalledResponder);
        let now = Now::blank(10, BodySense::default());
        let recall = RecallBundle {
            hits: vec![netherwick_now::RecallHit {
                frame_id: None,
                score: 0.7,
                summary: "remembered danger".to_string(),
                warning: Some("danger".to_string()),
                graph_context: Vec::new(),
            }],
            sense: MemorySense {
                place_danger: 0.9,
                similar_situation_count: 1,
                ..MemorySense::default()
            },
            first_person_summary: "I remember danger here.".to_string(),
            recollections: Vec::new(),
        };
        let ctx = EventContext {
            now: &now,
            latent: None,
            recall: Some(&recall),
            predicted_futures: &[],
            llm: None,
            safety: None,
        };
        let event = Event::new(10, EventKind::MemoryRecalled)
            .with_payload(EventPayload::MemoryRecalled { hits: 1 });

        let output = bus.dispatch(&ctx, &event).unwrap();

        assert!(output.iter().any(|response| matches!(
            response,
            Response::SetMemorySense(MemorySense { place_danger, .. }) if (*place_danger - 0.9).abs() < f32::EPSILON
        )));
    }

    #[test]
    fn safety_veto_responder_adds_experience_note() {
        let mut bus = EventBus::new();
        bus.on(responders::SafetyVetoResponder);
        let now = Now::blank(11, BodySense::default());
        let ctx = EventContext {
            now: &now,
            latent: None,
            recall: None,
            predicted_futures: &[],
            llm: None,
            safety: None,
        };
        let event =
            Event::new(11, EventKind::SafetyVetoed).with_payload(EventPayload::SafetyVetoed {
                desired_action: ActionPrimitive::Dock,
                reason: "Cliff".to_string(),
            });

        let output = bus.dispatch(&ctx, &event).unwrap();

        assert!(output.iter().any(|response| match response {
            Response::AddExperience(experience) => experience.text.contains("Cliff"),
            _ => false,
        }));
        assert!(output.iter().any(|response| matches!(
            response,
            Response::AddMemoryNote(note) if note.contains("Cliff")
        )));
    }

    #[test]
    fn reign_responder_maps_turn_to_action() {
        let mut bus = EventBus::new();
        bus.on(responders::ReignResponder);
        let now = Now::blank(10, BodySense::default());
        let ctx = EventContext {
            now: &now,
            latent: None,
            recall: None,
            predicted_futures: &[],
            llm: None,
            safety: None,
        };
        let input = ReignInput {
            id: Uuid::new_v4(),
            issued_at_ms: 10,
            expires_at_ms: 1_010,
            source: netherwick_actions::ReignSource::WebRemote,
            mode: netherwick_actions::ReignMode::Direct,
            command: netherwick_actions::ReignCommand::Turn {
                direction: TurnDir::Left,
                intensity: 0.5,
                duration_ms: 500,
            },
            priority: 1.0,
            note: None,
        };
        let event = Event::new(10, EventKind::ReignCommanded)
            .with_payload(EventPayload::ReignCommanded { input });

        let output = bus.dispatch(&ctx, &event).unwrap();

        assert!(output.iter().any(|response| {
            matches!(
                response,
                Response::ProposeAction(ActionPrimitive::Turn {
                    direction: TurnDir::Left,
                    intensity: 0.5,
                    duration_ms: 500,
                })
            )
        }));
    }

    fn test_reign_input(issued_at_ms: TimeMs) -> ReignInput {
        ReignInput {
            id: Uuid::new_v4(),
            issued_at_ms,
            expires_at_ms: issued_at_ms + 1_000,
            source: netherwick_actions::ReignSource::WebRemote,
            mode: netherwick_actions::ReignMode::Direct,
            command: netherwick_actions::ReignCommand::Stop,
            priority: 1.0,
            note: None,
        }
    }
}
