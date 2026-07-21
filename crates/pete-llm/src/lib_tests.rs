
use super::*;
use pete_actions::{ReignCommand, ReignInput, ReignMode, ReignSource};
use pete_body::{BodySense, CliffSensors};
use pete_now::{CapabilityAvailability, CapabilityBelief, CapabilityId, CapabilityKind, Now};
use uuid::Uuid;

struct SlowSceneProvider;

struct ImmediateSceneProvider;

#[async_trait]
impl CognitiveProvider for SlowSceneProvider {
    fn descriptor(&self) -> CognitiveProviderDescriptor {
        CognitiveProviderDescriptor {
            provider_id: ProviderId("slow-scene".to_string()),
            role: CognitiveRole::CognitiveAccelerator,
            implementation: "slow-fixture".to_string(),
            implementation_version: "1".to_string(),
            capabilities: vec![CapabilityDescriptor {
                capability: CognitiveCapability::DescribeScene,
                version: "1".to_string(),
                performance_confidence: 1.0,
                ..CapabilityDescriptor::default()
            }],
            health: ProviderHealth {
                state: ProviderHealthState::Available,
                confidence: 1.0,
                observed_at_ms: 0,
                valid_until_ms: u64::MAX,
                reason: None,
            },
            latency: LatencyEstimate {
                expected_ms: 1,
                p95_ms: 500,
            },
            locality: Locality::LocalNetwork,
            ..CognitiveProviderDescriptor::default()
        }
    }

    async fn execute(&mut self, _request: &CognitiveRequest) -> Result<CognitiveResponse> {
        tokio::time::sleep(Duration::from_millis(500)).await;
        anyhow::bail!("fixture disconnected")
    }
}

#[async_trait]
impl CognitiveProvider for ImmediateSceneProvider {
    fn descriptor(&self) -> CognitiveProviderDescriptor {
        CognitiveProviderDescriptor {
            provider_id: ProviderId("immediate-scene".to_string()),
            role: CognitiveRole::CognitiveAccelerator,
            implementation: "immediate-fixture".to_string(),
            implementation_version: "1".to_string(),
            capabilities: vec![CapabilityDescriptor {
                capability: CognitiveCapability::DescribeScene,
                version: "1".to_string(),
                performance_confidence: 1.0,
                ..CapabilityDescriptor::default()
            }],
            health: ProviderHealth {
                state: ProviderHealthState::Available,
                confidence: 1.0,
                observed_at_ms: 0,
                valid_until_ms: u64::MAX,
                reason: None,
            },
            latency: LatencyEstimate {
                expected_ms: 1,
                p95_ms: 1,
            },
            locality: Locality::LocalNetwork,
            ..CognitiveProviderDescriptor::default()
        }
    }

    async fn execute(&mut self, request: &CognitiveRequest) -> Result<CognitiveResponse> {
        Ok(CognitiveResponse {
            schema_version: 1,
            request_id: request.request_id.clone(),
            provider_id: ProviderId("immediate-scene".to_string()),
            provider_role: CognitiveRole::CognitiveAccelerator,
            implementation: "immediate-fixture".to_string(),
            implementation_version: "1".to_string(),
            model_version: Some("fixture-model".to_string()),
            status: CognitiveResponseStatus::Completed,
            confidence: 1.0,
            uncertainty: 0.0,
            input_snapshot: request.input_snapshot.clone(),
            completed_at_ms: request.created_at_ms.saturating_add(1),
            resource_cost: ResourceCost::default(),
            provenance: request.provenance.evidence_refs.clone(),
            payload: CognitiveResponsePayload::SceneDescription {
                text: "I see the fixture.".to_string(),
                embedding: vec![1.0],
            },
            failure: None,
        })
    }
}

fn tiny_eye_frame() -> EyeFrame {
    EyeFrame {
        captured_at_ms: 10,
        width: 1,
        height: 1,
        format: EyeFrameFormat::Rgb8,
        bytes: vec![1, 2, 3],
        source: Some("fixture".to_string()),
    }
}

#[test]
fn default_llm_config_uses_local_ollama() {
    let config = LlmConfig::default();

    assert_eq!(config.provider, LlmProvider::Ollama);
    assert_eq!(config.endpoint, "http://127.0.0.1:11434");
}

#[tokio::test]
async fn no_accelerator_scene_path_returns_without_blocking() {
    let mut cognition = LiveImageCognition::new(None);
    let started = std::time::Instant::now();
    let tick = cognition
        .poll_and_submit(Some(&tiny_eye_frame()), 1, 10)
        .await;
    assert!(started.elapsed() < Duration::from_millis(50));
    assert!(tick.registry.providers.is_empty());
    assert!(tick.enrichment.is_none());
}

#[tokio::test]
async fn slow_accelerator_never_blocks_organism_tick() {
    let mut router = CognitiveRouter::default();
    router.register(Box::new(SlowSceneProvider));
    let mut cognition = LiveImageCognition::from_router(router, 1_000);
    let started = std::time::Instant::now();
    let tick = cognition
        .poll_and_submit(Some(&tiny_eye_frame()), 1, 10)
        .await;
    assert!(started.elapsed() < Duration::from_millis(50));
    assert_eq!(tick.registry.providers.len(), 1);
    assert!(tick.enrichment.is_none());
}

#[tokio::test]
async fn completed_response_for_replaced_frame_is_stale() {
    let mut router = CognitiveRouter::default();
    router.register(Box::new(ImmediateSceneProvider));
    let mut cognition = LiveImageCognition::from_router(router, 1_000);
    cognition
        .poll_and_submit(Some(&tiny_eye_frame()), 1, 10)
        .await;
    tokio::task::yield_now().await;

    let mut replacement = tiny_eye_frame();
    replacement.captured_at_ms = 11;
    replacement.bytes = vec![3, 2, 1];
    let tick = cognition.poll_and_submit(Some(&replacement), 2, 11).await;

    let response = tick.response.expect("completed response");
    assert_eq!(response.disposition, ResponseDisposition::Stale);
    assert_eq!(response.response.status, CognitiveResponseStatus::Stale);
    assert!(tick.enrichment.is_none());
}

#[test]
fn self_context_never_lists_unavailable_capability_as_available() {
    let mut now = Now::blank(10, BodySense::default());
    let id = CapabilityId("sensor:vision".to_string());
    now.world.self_model.capabilities.capabilities.insert(
        id.clone(),
        CapabilityBelief {
            id,
            kind: CapabilityKind::Sensor,
            availability: CapabilityAvailability::Unavailable,
            confidence: 1.0,
            unavailable_reason: Some("camera is unplugged".to_string()),
            ..CapabilityBelief::default()
        },
    );
    let rendered = render_self_model_context(&now);
    assert!(rendered.contains("available=[]"));
    assert!(rendered.contains("unavailable=[sensor:vision (camera is unplugged)]"));
}

#[test]
fn extracts_json_from_fenced_response() {
    let text = "```json\n{\"summary\":\"hi\",\"confidence\":0.9}\n```";
    let json = extract_json_object(text).unwrap();
    assert_eq!(json, "{\"summary\":\"hi\",\"confidence\":0.9}");
}

#[test]
fn extracts_json_from_wrapped_response_text() {
    let text = "Sure, here you go:\n{\"summary\":\"hi\",\"confidence\":0.9}\nThanks";
    let json = extract_json_object(text).unwrap();
    assert_eq!(json, "{\"summary\":\"hi\",\"confidence\":0.9}");
}

#[test]
fn parses_turn_action() {
    let action = parse_action_spec(ActionSpec {
        kind: "turn".to_string(),
        direction: Some("left".to_string()),
        intensity: Some(0.6),
        duration_ms: Some(1200),
        target: None,
        text: None,
        style: None,
        pattern: None,
    })
    .unwrap();
    assert_eq!(
        action,
        ActionPrimitive::Turn {
            direction: TurnDir::Left,
            intensity: 0.6,
            duration_ms: 1200,
        }
    );
}

#[test]
fn parses_speak_action() {
    let action = parse_action_spec(ActionSpec {
        kind: "speak".to_string(),
        direction: None,
        intensity: None,
        duration_ms: None,
        target: None,
        text: Some("hello from the llm".to_string()),
        style: None,
        pattern: None,
    })
    .unwrap();

    assert_eq!(
        action,
        ActionPrimitive::Speak {
            text: "hello from the llm".to_string()
        }
    );
}

#[test]
fn parses_llm_json_explore_action_into_decision() {
    let decision = parse_llm_decision_json(r#"{"action":{"kind":"explore"}}"#, true).unwrap();
    assert_eq!(
        decision.action,
        Some(ActionPrimitive::Explore {
            style: ExploreStyle::RandomWalk,
            duration_ms: 1_000,
        })
    );
}

#[test]
fn commands_disabled_ignores_llm_json_action() {
    let decision = parse_llm_decision_json(r#"{"action":{"kind":"explore"}}"#, false).unwrap();
    assert_eq!(decision.action, None);
}

#[test]
fn summarized_senses_include_latest_reign_command() {
    let mut now = Now::blank(100, BodySense::default());
    now.reign.latest = Some(ReignInput {
        id: Uuid::new_v4(),
        issued_at_ms: 100,
        expires_at_ms: 1_100,
        source: ReignSource::WebRemote,
        mode: ReignMode::Direct,
        command: ReignCommand::Turn {
            direction: TurnDir::Left,
            intensity: 0.5,
            duration_ms: 500,
        },
        priority: 1.0,
        note: Some("turn toward charger".to_string()),
    });
    now.reign.active = true;
    now.reign.mode = Some(ReignMode::Direct);

    let senses = summarized_senses(&now).join("\n");

    assert!(senses.contains("Remote control active: Direct"));
    assert!(senses.contains("Latest remote command: Turn Left"));
    assert!(senses.contains(
            "Matching executable remote action JSON: {\"direction\":\"left\",\"duration_ms\":500,\"intensity\":0.5,\"kind\":\"turn\"}."
        ));
    assert!(senses.contains("Remote note: turn toward charger"));
}

#[test]
fn active_reign_action_becomes_llm_command_action() {
    let command = ReignCommand::Go {
        intensity: 0.4,
        duration_ms: 700,
    };
    let mut now = Now::blank(100, BodySense::default());
    now.reign.active = true;
    now.reign.mode = Some(ReignMode::Assist);
    now.reign.latest = Some(ReignInput {
        id: Uuid::new_v4(),
        issued_at_ms: 100,
        expires_at_ms: 1_100,
        source: ReignSource::WebRemote,
        mode: ReignMode::Assist,
        command: command.clone(),
        priority: 1.0,
        note: None,
    });

    assert_eq!(active_reign_action(&now), command.to_action());
    assert_eq!(
        reign_command_summary(&now),
        Some("Following Reign command: Go, intensity 0.40, 700ms".to_string())
    );
}

#[test]
fn observe_only_reign_does_not_become_llm_command_action() {
    let mut now = Now::blank(100, BodySense::default());
    now.reign.active = true;
    now.reign.mode = Some(ReignMode::ObserveOnly);
    now.reign.latest = Some(ReignInput {
        id: Uuid::new_v4(),
        issued_at_ms: 100,
        expires_at_ms: 1_100,
        source: ReignSource::WebRemote,
        mode: ReignMode::ObserveOnly,
        command: ReignCommand::Stop,
        priority: 1.0,
        note: None,
    });
    assert_eq!(active_reign_action(&now), None);
}

#[test]
fn agent_prompt_frames_actions_as_executable_and_reign_as_command_input() {
    let now = Now::blank(100, BodySense::default());
    let prompt = build_agent_prompt(
        &now,
        None,
        &ExperienceLatent::default(),
        &[],
        "none",
        Some("I am awake."),
        &LlmConfig::default(),
    );

    assert!(prompt.contains("choose a high-level action primitive"));
    assert!(prompt.contains("executable command candidate"));
    assert!(prompt.contains("not only a suggestion or note"));
    assert!(prompt.contains("Never output raw motor control such as wheel speeds"));
    assert!(prompt.contains("Treat Reign controls as present-tense command input"));
    assert!(prompt.contains("set action to the matching allowed action"));
    assert!(prompt.contains("Allowed chirp patterns and meanings"));
    assert!(prompt.contains("goal_acquired: notes 72,79,84,91"));
    assert!(prompt.contains("person_recognized: notes 76,79,84,79"));
    assert!(!prompt.contains("Do not override active Direct reign"));
}

#[test]
fn prompts_offer_generated_uuid_options_instead_of_asking_llm_to_invent_ids() {
    let now = Now::blank(100, BodySense::default());
    let prompt = build_agent_prompt(
        &now,
        None,
        &ExperienceLatent::default(),
        &[],
        "none",
        Some("I am awake."),
        &LlmConfig::default(),
    );

    assert!(prompt.contains("choose one of these exact UUID options"));
    assert!(prompt.contains("do not invent your own"));

    let options = prompt
        .lines()
        .skip_while(|line| !line.contains("choose one of these exact UUID options"))
        .skip(1)
        .take_while(|line| line.starts_with("- "))
        .map(|line| line.trim_start_matches("- "))
        .collect::<Vec<_>>();

    assert_eq!(options.len(), PROMPT_UUID_OPTION_COUNT);
    for option in options {
        Uuid::parse_str(option).expect("prompt UUID option should be valid");
    }
}

#[test]
fn summarized_senses_include_input_sensor_channels() {
    let mut now = Now::blank(100, BodySense::default());
    now.body.infrared_character = 248;
    now.body.flags.cliff_front_left = true;
    now.body.flags.wall = true;
    now.body.flags.virtual_wall = true;
    now.body.cliff_sensors = CliffSensors {
        left: 0.10,
        front_left: 0.80,
        front_right: 0.40,
        right: 0.20,
    };
    now.kinect.ir = vec![0.1, 0.8, 0.9, 0.2];

    let senses = summarized_senses(&now).join("\n");

    assert!(senses.contains("I feel the floor fall away near me."));
    assert!(!senses.contains("Cliff sensor levels"));
    assert!(senses.contains("My wall sensor is active."));
    assert!(senses.contains("I detect a virtual wall."));
    assert!(senses.contains("My Create IR receiver reports character 248."));
    assert!(senses.contains("Kinect IR has 4 samples, mean 0.50, max 0.90, bright fraction 0.50."));
}

#[test]
fn combobulator_prompt_uses_timeline_distillation_rules() {
    let mut now = Now::blank(250, BodySense::default());
    now.ear.transcript = Some("hello there".to_string());

    let impression = Impression::new(
        "audio.transcript.impression",
        "I hear: <hello there>",
        Vec::new(),
        now.t_ms,
        now.t_ms,
    )
    .with_confidence(0.8)
    .with_payload(serde_json::json!({
        "generator": "mechanical",
        "faculty": "ear.mechanical_impression",
    }));
    let prompt = build_combobulator_prompt(
        &now,
        &[impression],
        None,
        &ExperienceLatent::default(),
        &[],
        "I remember Tim.",
    );

    assert!(prompt.contains("Timeline evidence:"));
    assert!(prompt.contains("[T+00.000 - T+00.000 | "));
    assert!(prompt.contains("IMPRESSION id="));
    assert!(prompt.contains("kind=audio.transcript.impression"));
    assert!(prompt.contains("generator=\"mechanical\""));
    assert!(prompt.contains("faculty=\"ear.mechanical_impression\""));
    assert!(prompt.contains("confidence=0.800"));
    assert!(prompt.contains("occurred_at="));
    assert!(prompt.contains("observed_at="));
    assert!(prompt.contains(".250"));
    assert!(prompt.contains(":00 to "));
    assert!(prompt.contains("what is going on right now"));
    assert!(prompt.contains("first-person lived experience"));
    assert!(
        prompt.contains("Convert raw body data into feeling-centered first-person interpretations")
    );
    assert!(prompt.contains("I feel steady"));
    assert!(prompt.contains("telling someone with amnesia"));
    assert!(prompt.contains("Distill what matters, not what the records said."));
    assert!(prompt.contains("Treat the entries as fragmentary, possibly contradictory"));
    assert!(prompt.contains("not as the topic to describe"));
    assert!(prompt.contains("do not group by faculty or source"));
    assert!(prompt.contains("Do not infer emotional tone"));
    assert!(prompt.contains("do not enumerate ids"));
    assert!(prompt.contains("Do not assume a human is currently present"));
    assert!(prompt.contains("CONTEXT FRAME"));
    assert!(prompt.contains("text=\"I hear: \\u003chello there\\u003e\""));
}

#[test]
fn scientific_review_prompt_frames_training_rows_as_uncertain_evidence() {
    let request = LlmReviewRequest::training_example(
        42,
        LlmTrainingExampleEvidence {
            example_id: "danger-row-7".to_string(),
            behavior: "danger".to_string(),
            input_summary: "front range says clear; no bump flags".to_string(),
            expected_summary: "bump_risk=1.0".to_string(),
            actual_summary: Some("model predicted low bump risk".to_string()),
            reward: Some(-0.2),
            source: Some("world_outcome".to_string()),
            contradictions: vec!["no contact evidence supports bump label".to_string()],
            missing_evidence: vec!["no post-action body flags".to_string()],
        },
    );

    let prompt = build_scientific_review_prompt(&request);

    assert!(prompt.contains("scientific critic, not its source of truth"));
    assert!(prompt.contains("must not declare identity as certain"));
    assert!(prompt.contains("mark a training row as true"));
    assert!(prompt.contains("suspicious_training_examples"));
    assert!(prompt.contains("label_proposals"));
    assert!(prompt.contains("\"example_id\": \"danger-row-7\""));
    assert!(prompt.contains("no contact evidence supports bump label"));
    assert!(prompt.contains("AVAILABLE ACTIONS JSON\n- none"));
}

#[test]
fn parse_scientific_review_json_clamps_and_reuses_existing_action_types() {
    let request = LlmReviewRequest::training_example(
        99,
        LlmTrainingExampleEvidence {
            example_id: "charge-row-3".to_string(),
            behavior: "charge".to_string(),
            input_summary: "battery dropping, charger not visible".to_string(),
            expected_summary: "charging_started=true".to_string(),
            ..LlmTrainingExampleEvidence::default()
        },
    );
    let review = parse_scientific_review_json(
            &request,
            r#"Here is JSON:
{
  "critique":"Label is plausible but unsupported by visible charger or battery delta.",
  "counterfactuals":[{"instead_of":null,"proposed":{"kind":"inspect","target":"charger"},"reason":"Look for missing charger evidence.","weight":1.5}],
  "suggested_tests":[{"action":{"kind":"inspect","target":"charger"},"question":"Is the charger visible?","expected_observation":"charger appears","disconfirming_observation":"no charger evidence","risk_note":"","confidence":-0.2}],
  "suspicious_training_examples":[{"example_id":"","reason":"Expected charging label lacks support.","severity":2.0,"suspected_issue":"unsupported_label","supporting_evidence":["charger not visible"],"missing_evidence":["battery delta"],"suggested_fix":"send to human review"}],
  "label_proposals":[{"example_id":"","proposed_label":"charging_started=unknown","rationale":"evidence is incomplete","confidence":0.7,"requires_human_review":true}],
  "human_review_prompts":["Check whether charging actually started."],
  "confidence":1.2
}"#,
        )
        .expect("scientific review json should parse");

    assert_eq!(review.t_ms, 99);
    assert_eq!(review.target_id, "charge-row-3");
    assert_eq!(review.target_kind, ReviewTargetKind::TrainingExample);
    assert_eq!(review.confidence, 1.0);
    assert_eq!(review.counterfactuals.len(), 1);
    assert_eq!(review.counterfactuals[0].weight, 1.0);
    assert_eq!(
        review.suggested_tests[0].action,
        Some(ActionPrimitive::Inspect {
            target: InspectTarget::Charger
        })
    );
    assert_eq!(review.suggested_tests[0].confidence, 0.0);
    assert_eq!(
        review.suspicious_training_examples[0].example_id,
        "charge-row-3"
    );
    assert_eq!(review.suspicious_training_examples[0].severity, 1.0);
    assert_eq!(review.label_proposals[0].confidence, 0.7);
    assert!(review.label_proposals[0].requires_human_review);
}

#[test]
fn stream_line_uses_thinking_when_response_is_whitespace() {
    let mut rx = subscribe_llm_streams();
    while rx.try_recv().is_ok() {}

    let mut body = String::new();
    handle_ollama_stream_line(
        7,
        "combobulator",
        "gpt-oss:20b",
        r#"{"response":"\n","thinking":"hello","done":false}"#,
        &mut body,
    )
    .expect("stream line should parse");

    let event = next_stream_event_for_id(&mut rx, 7);
    assert_eq!(event.phase, LlmStreamPhase::Delta);
    assert_eq!(event.delta.as_deref(), Some("hello"));
    assert_eq!(body, "\n");
}

#[test]
fn stream_line_prefers_response_when_non_whitespace() {
    let mut rx = subscribe_llm_streams();
    while rx.try_recv().is_ok() {}

    let mut body = String::new();
    handle_ollama_stream_line(
        8,
        "agent",
        "llama3.2",
        r#"{"response":"ok","thinking":"ignored","done":false}"#,
        &mut body,
    )
    .expect("stream line should parse");

    let event = next_stream_event_for_id(&mut rx, 8);
    assert_eq!(event.phase, LlmStreamPhase::Delta);
    assert_eq!(event.delta.as_deref(), Some("ok"));
    assert_eq!(body, "ok");
}

fn next_stream_event_for_id(
    rx: &mut tokio::sync::broadcast::Receiver<LlmStreamEvent>,
    id: u64,
) -> LlmStreamEvent {
    for _ in 0..64 {
        let event = rx.try_recv().expect("delta event should be emitted");
        if event.id == id {
            return event;
        }
    }
    panic!("delta event for stream {id} was not emitted");
}

#[test]
fn prompts_include_embodied_context_without_raw_vectors() {
    let sensation_id = Uuid::new_v4();
    let experience_id = Uuid::new_v4();
    let context = EmbodiedContext {
        experience_id: Some(experience_id),
        summary: "I see a frame and focus on part of it.".to_string(),
        sensations: vec![pete_experience::EmbodiedSensationRef {
            id: sensation_id,
            parent_id: Some(Uuid::new_v4()),
            modality: pete_experience::Modality::Vision,
            payload_kind: pete_experience::SensationPayloadKind::ImageBytes,
            kind: "vision.image_bytes".to_string(),
            source: "camera".to_string(),
            summary: Some("A camera frame is visible.".to_string()),
        }],
        impressions: Vec::new(),
        lineage: Vec::new(),
        sensation_vectors: Vec::new(),
        impression_vectors: Vec::new(),
        predictions: Vec::new(),
        memory_links: Vec::new(),
    };
    let now = Now::blank(100, BodySense::default());

    let prompt = build_agent_prompt(
        &now,
        Some(&context),
        &ExperienceLatent::default(),
        &[],
        "none",
        None,
        &LlmConfig::default(),
    );

    assert!(prompt.contains("Current embodied experience:"));
    assert!(prompt.contains(&format!("experience_id: {experience_id}")));
    assert!(prompt.contains("derived_sensations=1"));
    assert!(prompt.contains("payload=image_bytes"));
    assert!(!prompt.contains("[0."));
}

#[test]
fn image_caption_prompt_frames_live_vision_viewpoint() {
    assert!(IMAGE_CAPTION_PROMPT.contains("Describe only what you see from your viewpoint"));
    assert!(IMAGE_CAPTION_PROMPT.contains("your own vision looking out"));
    assert!(IMAGE_CAPTION_PROMPT.contains("not that visible people"));
    assert!(IMAGE_CAPTION_PROMPT.contains("the machine's own live view"));
    assert!(IMAGE_CAPTION_PROMPT.contains("When looking out, one does not see oneself"));
    assert!(IMAGE_CAPTION_PROMPT.contains("most likely someone you're looking at, not yourself"));
    assert!(
        IMAGE_CAPTION_PROMPT.contains("unless you're clearly looking in a mirror or reflection")
    );
    assert!(IMAGE_CAPTION_PROMPT.contains("Describe visible people in third person"));
    assert!(!IMAGE_CAPTION_PROMPT.contains("data:image"));
}

#[test]
fn heuristic_combobulation_prefers_concrete_present_evidence() {
    let mut now = Now::blank(500, BodySense::default());
    now.ear.transcript = Some("come over here".to_string());
    now.body.flags.bump_left = true;

    let combobulation = heuristic_combobulation(&now, "A stale memory.");

    assert_eq!(combobulation.summary, "I hear: come over here");
    assert_eq!(combobulation.confidence, 0.35);
}
