use std::collections::VecDeque;

use anyhow::Result;
use netherwick_actions::{ActionPrimitive, ApproachTarget, TurnDir};
use netherwick_autonomic::SafetyReason;
use netherwick_core::{Provenance, TimeMs};
use netherwick_experience::{Experience, ExperienceLatent, Impression, FuturePrediction, Sensation};
use netherwick_llm::LlmTeaching;
use netherwick_memory::RecallBundle;
use netherwick_now::{MemorySense, Now};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use uuid::Uuid;

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
    MemoryRecalled,
    PredictionMade,
    SurpriseHigh,
    LlmCommanded,
    LlmCritiqued,
    SafetyVetoed,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Event {
    pub id: Uuid,
    pub t_ms: TimeMs,
    pub kind: EventKind,
    pub summary: Option<String>,
    pub provenance: Provenance,
    pub payload: Value,
}

impl Event {
    pub fn new(t_ms: TimeMs, kind: EventKind) -> Self {
        Self {
            id: Uuid::new_v4(),
            t_ms,
            kind,
            summary: None,
            provenance: Provenance::direct().with_stage("event"),
            payload: Value::Null,
        }
    }

    pub fn with_summary(mut self, summary: impl Into<String>) -> Self {
        self.summary = Some(summary.into());
        self
    }

    pub fn with_payload(mut self, payload: Value) -> Self {
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
    None,
    AddSensation(Sensation),
    AddImpression(Impression),
    AddExperience(Experience),
    ProposeAction(ActionPrimitive),
    SetDrive { name: String, value: f32 },
    SetMemorySense(MemorySense),
    Teach(LlmTeaching),
    RememberNote(String),
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
                if let Response::Emit(next) = response {
                    queue.push_back(next.clone());
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
}

impl Default for EventExtractorConfig {
    fn default() -> Self {
        Self {
            low_battery_threshold: 0.20,
            face_familiarity_threshold: 0.70,
            surprise_threshold: 0.75,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct EventExtractor {
    pub config: EventExtractorConfig,
    last_now: Option<Now>,
}

impl EventExtractor {
    pub fn events_from_now(&mut self, now: &Now, recall: Option<&RecallBundle>) -> Vec<Event> {
        let mut events = Vec::new();
        let previous = self.last_now.as_ref();

        if now.body.battery_level <= self.config.low_battery_threshold {
            events.push(
                Event::new(now.t_ms, EventKind::BatteryLow)
                    .with_summary(format!("Battery low at {:.2}.", now.body.battery_level))
                    .with_payload(json!({ "battery_level": now.body.battery_level })),
            );
        }

        if now.body.charging && previous.map(|prior| !prior.body.charging).unwrap_or(false) {
            events.push(
                Event::new(now.t_ms, EventKind::ChargingStarted)
                    .with_summary("Charging started.")
                    .with_payload(json!({ "battery_level": now.body.battery_level })),
            );
        }

        if !now.body.charging && previous.map(|prior| prior.body.charging).unwrap_or(false) {
            events.push(Event::new(now.t_ms, EventKind::ChargingStopped).with_summary("Charging stopped."));
        }

        if now.body.flags.bump_left || now.body.flags.bump_right {
            let side = bump_side(now);
            events.push(
                Event::new(now.t_ms, EventKind::Bump)
                    .with_summary(format!("Bump on the {side} side."))
                    .with_payload(json!({ "side": side })),
            );
        }

        if now.body.flags.cliff_left || now.body.flags.cliff_right {
            events.push(Event::new(now.t_ms, EventKind::Cliff).with_summary("Cliff detected."));
        }

        if now.body.flags.wheel_drop {
            events.push(Event::new(now.t_ms, EventKind::WheelDrop).with_summary("Wheel drop detected."));
        }

        if let Some(transcript) = &now.ear.transcript {
            if !transcript.trim().is_empty() {
                events.push(
                    Event::new(now.t_ms, EventKind::SpeechHeard)
                        .with_summary(format!("Speech heard: {}", transcript.trim()))
                        .with_payload(json!({ "transcript": transcript.trim() })),
                );
            }
        }

        let face_familiarity = recall
            .map(|bundle| bundle.sense.face_familiarity)
            .unwrap_or(now.memory.face_familiarity);
        if !now.face.embeddings.is_empty() && face_familiarity >= self.config.face_familiarity_threshold {
            events.push(
                Event::new(now.t_ms, EventKind::FaceRecognized)
                    .with_summary("Recognized a familiar face.")
                    .with_payload(json!({
                        "face_embeddings": now.face.embeddings.len(),
                        "face_familiarity": face_familiarity,
                    })),
            );
        }

        if let Some(bundle) = recall {
            if !bundle.hits.is_empty() {
                events.push(
                    Event::new(now.t_ms, EventKind::MemoryRecalled)
                        .with_summary(format!("Recalled {} memory hits.", bundle.hits.len()))
                        .with_payload(json!({ "hits": bundle.hits.len() })),
                );
            }
        }

        if now.surprise.total >= self.config.surprise_threshold {
            events.push(
                Event::new(now.t_ms, EventKind::SurpriseHigh)
                    .with_summary(format!("Surprise high at {:.2}.", now.surprise.total))
                    .with_payload(json!({ "total": now.surprise.total })),
            );
        }

        self.last_now = Some(now.clone());
        events
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
    payload: Value,
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
                Response::RememberNote("Battery low here.".to_string()),
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
                    "I taste charging.",
                    json!({ "charging": true }),
                )),
                Response::AddExperience(
                    Experience::new(
                        "body.charging",
                        "Charging happened here.",
                        Vec::new(),
                        Vec::new(),
                        event.t_ms,
                        event.t_ms,
                    ),
                ),
                Response::SetDrive {
                    name: "battery_hunger".to_string(),
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
            let side = event.payload.get("side").and_then(Value::as_str).unwrap_or("unknown");
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
                Response::RememberNote("Danger here.".to_string()),
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
            let mut responses = Vec::new();
            for recollection in &recall.recollections {
                responses.push(Response::AddSensation(recollection.sensation.clone()));
            }
            responses.push(Response::SetMemorySense(recall.sense.clone()));
            responses.push(Response::AddSensation(event_sensation(
                event,
                "memory.recall",
                "memory",
                recall.first_person_summary.clone(),
                json!({ "hits": recall.hits.len() }),
            )));
            Ok(responses)
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
            let reason = event
                .payload
                .get("reason")
                .and_then(Value::as_str)
                .unwrap_or("unknown reason");
            Ok(vec![
                Response::AddExperience(Experience::new(
                    "safety.veto",
                    format!("My body refused the command because {reason}."),
                    Vec::new(),
                    Vec::new(),
                    event.t_ms,
                    event.t_ms,
                )),
                Response::Teach(LlmTeaching {
                    t_ms: event.t_ms,
                    summary: format!("Safety vetoed an action because {reason}."),
                    critique: Some("Body safety overrode the chosen action.".to_string()),
                    counterfactuals: Vec::new(),
                    memory_notes: vec![format!("Avoid repeating action after safety veto: {reason}")],
                    confidence: 0.9,
                }),
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
                    target: netherwick_actions::InspectTarget::Person,
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
    bus.on(responders::BatteryLowResponder);
    bus.on(responders::ChargingStartedResponder);
    bus.on(responders::BumpResponder);
    bus.on(responders::MemoryRecalledResponder);
    bus.on(responders::SafetyVetoResponder);
    bus.on(responders::FaceRecognizedResponder);
    bus
}

pub fn safety_veto_event(
    now: &Now,
    desired: &ActionPrimitive,
    reason: Option<SafetyReason>,
) -> Event {
    let reason_text = reason
        .map(|value| format!("{value:?}"))
        .unwrap_or_else(|| "Unknown".to_string());
    Event::new(now.t_ms, EventKind::SafetyVetoed)
        .with_summary(format!("Safety vetoed {:?}.", desired))
        .with_payload(json!({
            "desired_action": desired,
            "reason": reason_text,
        }))
        .with_provenance(Provenance::direct().with_stage("safety"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use netherwick_body::BodySense;
    use netherwick_memory::RecallBundle;

    #[test]
    fn extractor_emits_low_battery_and_speech() {
        let mut extractor = EventExtractor::default();
        let mut body = BodySense::default();
        body.battery_level = 0.1;
        let mut now = Now::blank(42, body);
        now.ear.transcript = Some("hello".to_string());

        let events = extractor.events_from_now(&now, None);

        assert!(events.iter().any(|event| event.kind == EventKind::BatteryLow));
        assert!(events.iter().any(|event| event.kind == EventKind::SpeechHeard));
    }

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
            }],
            ..RecallBundle::default()
        };

        let events = extractor.events_from_now(&now, Some(&recall));

        assert!(events.iter().any(|event| event.kind == EventKind::MemoryRecalled));
    }

    #[test]
    fn battery_low_responder_proposes_dock() {
        let mut bus = EventBus::new();
        bus.on(responders::BatteryLowResponder);
        let now = Now::blank(5, BodySense::default());
        let ctx = EventContext {
            now: &now,
            latent: None,
            recall: None,
            predicted_futures: &[],
        };
        let event = Event::new(5, EventKind::BatteryLow);

        let output = bus.dispatch(&ctx, &event).unwrap();

        assert!(output
            .iter()
            .any(|response| matches!(response, Response::ProposeAction(ActionPrimitive::Dock))));
    }
}
