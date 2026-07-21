#[test]
fn possessor_skill_runtime_turns_and_completes_from_updated_target_error() {
    let mut cockpit = SimCockpit::new().with_unscoped_bench_mode();
    let events = cockpit.get_events_since(0).unwrap();
    let mut runtime = PossessorSkillRuntime::default();
    let request = SkillRequest {
        skill_id: SkillId::TurnTowardTarget,
        target: Some(pete_now::EntityId("charger:17".to_string())),
        bearing_rad: Some(0.5),
        maximum_duration_ms: 1_000,
        expected_progress: 0.8,
        ..SkillRequest::default()
    };

    let body = BodySense::default();
    let _ = possessor_test_step(
        &mut runtime,
        &mut cockpit,
        &request,
        &body,
        false,
        &events,
        1,
    );
    let (running, command_sent) = possessor_test_step(
        &mut runtime,
        &mut cockpit,
        &request,
        &body,
        false,
        &events,
        101,
    );
    assert!(command_sent);
    assert_eq!(running.phase, SkillPhase::Running);
    assert_eq!(running.attempts, 1);
    assert_eq!(running.dispatch_count, 1);
    assert_ne!(running.execution_id, 0);

    let aligned = SkillRequest {
        bearing_rad: Some(0.05),
        ..request
    };
    let (completed, stop_sent) = possessor_test_step(
        &mut runtime,
        &mut cockpit,
        &aligned,
        &body,
        false,
        &events,
        201,
    );
    assert!(stop_sent);
    assert_eq!(completed.phase, SkillPhase::Terminal);
    assert_eq!(completed.outcome, Some(SkillOutcome::Completed));
    assert_eq!(completed.progress, Some(1.0));
    assert_eq!(completed.execution_id, running.execution_id);
    assert_eq!(completed.attempts, 1);
}

#[test]
fn possessor_terminal_status_is_consumed_once_before_a_real_retry() {
    let mut cockpit = SimCockpit::new().with_unscoped_bench_mode();
    let events = cockpit.get_events_since(0).unwrap();
    let mut runtime = PossessorSkillRuntime::default();
    let request = SkillRequest {
        skill_id: SkillId::ApproachTarget,
        goal_id: Some(pete_conductor::GoalId::new("seek_charger")),
        behavior_id: Some("approach_charger".to_string()),
        target: Some(pete_now::EntityId("charger:17".to_string())),
        bearing_rad: Some(0.0),
        range_m: Some(2.0),
        stop_range_m: Some(0.3),
        maximum_duration_ms: 100,
        ..SkillRequest::default()
    };

    let body = BodySense::default();
    let _ = possessor_test_step(
        &mut runtime,
        &mut cockpit,
        &request,
        &body,
        false,
        &events,
        1,
    );
    let (running, first_dispatch) = possessor_test_step(
        &mut runtime,
        &mut cockpit,
        &request,
        &body,
        false,
        &events,
        51,
    );
    assert!(first_dispatch);
    assert_eq!(running.attempts, 1);
    assert_eq!(running.dispatch_count, 1);

    let (timed_out, stop_sent) = possessor_test_step(
        &mut runtime,
        &mut cockpit,
        &request,
        &body,
        false,
        &events,
        101,
    );
    assert!(stop_sent);
    assert_eq!(timed_out.phase, SkillPhase::Terminal);
    assert_eq!(timed_out.outcome, Some(SkillOutcome::TimedOut));
    assert_eq!(timed_out.execution_id, running.execution_id);
    assert_eq!(timed_out.attempts, 1);

    let _ = possessor_test_step(
        &mut runtime,
        &mut cockpit,
        &request,
        &body,
        false,
        &events,
        102,
    );
    let (retry, retry_dispatched) = possessor_test_step(
        &mut runtime,
        &mut cockpit,
        &request,
        &body,
        false,
        &events,
        152,
    );
    assert!(retry_dispatched);
    assert_eq!(retry.phase, SkillPhase::Running);
    assert_ne!(retry.execution_id, timed_out.execution_id);
    assert_eq!(retry.attempts, 2);
    assert_eq!(retry.dispatch_count, 1);
}

#[test]
fn possessor_motor_refreshes_do_not_increment_attempts() {
    let mut cockpit = SimCockpit::new().with_unscoped_bench_mode();
    let events = cockpit.get_events_since(0).unwrap();
    let mut runtime = PossessorSkillRuntime::default();
    let request = SkillRequest {
        skill_id: SkillId::FollowBearing,
        bearing_rad: Some(0.5),
        maximum_duration_ms: 5_000,
        ..SkillRequest::default()
    };

    let body = BodySense::default();
    let (first, _) = possessor_test_step(
        &mut runtime,
        &mut cockpit,
        &request,
        &body,
        false,
        &events,
        1,
    );
    let (second, _) = possessor_test_step(
        &mut runtime,
        &mut cockpit,
        &request,
        &body,
        false,
        &events,
        101,
    );
    let (third, _) = possessor_test_step(
        &mut runtime,
        &mut cockpit,
        &request,
        &body,
        false,
        &events,
        201,
    );

    assert_eq!(first.execution_id, second.execution_id);
    assert_eq!(second.execution_id, third.execution_id);
    assert_eq!(third.attempts, 1);
    assert_eq!(third.dispatch_count, 2);
}

#[test]
fn lua_skill_progress_preserves_goal_metric_and_normalizes_from_baseline() {
    let mut cockpit = SimCockpit::new().with_unscoped_bench_mode();
    let events = cockpit.get_events_since(0).unwrap();
    let mut runtime = PossessorSkillRuntime::default();
    let request = SkillRequest {
        skill_id: SkillId::ApproachTarget,
        goal_id: Some(pete_conductor::GoalId::new("seek_charger")),
        target: Some(pete_now::EntityId("charger:17".to_string())),
        bearing_rad: Some(0.0),
        range_m: Some(2.0),
        stop_range_m: Some(0.3),
        maximum_duration_ms: 5_000,
        progress_metric: "target_distance".to_string(),
        progress_baseline: Some(2.0),
        progress_tolerance: 0.1,
        ..SkillRequest::default()
    };
    let body = BodySense::default();
    let _ = possessor_test_step(
        &mut runtime,
        &mut cockpit,
        &request,
        &body,
        false,
        &events,
        1,
    );
    let (started, _) = possessor_test_step(
        &mut runtime,
        &mut cockpit,
        &request,
        &body,
        false,
        &events,
        101,
    );
    assert_eq!(started.progress, Some(0.0));
    assert_eq!(started.request.goal_id, request.goal_id);
    assert_eq!(started.request.progress_metric, "target_distance");

    let halfway = SkillRequest {
        range_m: Some(1.0),
        ..request
    };
    let (progressed, _) = possessor_test_step(
        &mut runtime,
        &mut cockpit,
        &halfway,
        &body,
        false,
        &events,
        201,
    );
    assert_eq!(progressed.progress, Some(0.5));
    assert_eq!(
        progressed
            .script
            .as_ref()
            .map(|script| script.skill_id.as_str()),
        Some("motherbrain.approachTarget")
    );
    let provenance = runtime
        .provenance
        .as_ref()
        .expect("running skill provenance");
    assert_eq!(
        provenance
            .pointer("/request/goal_id")
            .and_then(serde_json::Value::as_str),
        Some("seek_charger")
    );
    assert_eq!(
        provenance
            .pointer("/diagnostics/progress/target_distance")
            .and_then(serde_json::Value::as_f64),
        Some(1.0)
    );
    let mut ledger_now = Now::blank(250, body);
    runtime.annotate_now(&mut ledger_now);
    assert!(ledger_now
        .extensions
        .contains_key("motherbrain.skill_execution"));
}

#[test]
fn recognized_person_greeting_runs_through_goal_lua_and_acknowledges_encounter() {
    std::thread::Builder::new()
        .name("lua-greet-goal-test".to_string())
        .stack_size(8 * 1024 * 1024)
        .spawn(|| {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(async {
                    let ledger = JsonlLedger::new("/tmp/pete-runtime-lua-greet-goal-test");
                    let memory = InMemoryExperienceStore::new();
                    let recall = FixedRecall {
                        bundle: RecallBundle {
                            sense: pete_now::MemorySense {
                                face_familiarity: 0.92,
                                remembered_entities: vec![pete_now::GraphEntity {
                                    id: "person:alex".to_string(),
                                    labels: vec!["Person".to_string()],
                                    summary: "Alex".to_string(),
                                    score: 0.90,
                                }],
                                ..pete_now::MemorySense::default()
                            },
                            ..RecallBundle::default()
                        },
                    };
                    let mut runtime = MinimalRuntime::new(
                        ledger,
                        memory,
                        recall,
                        FixedConductor::new(ActionPrimitive::Stop),
                        SimpleSafety::default(),
                        pete_llm::NoopLlmAgent,
                    )
                    .with_action_selector_mode(ActionSelectorMode::Goal);
                    let person_now = |t_ms| {
                        let mut now = idle_now(t_ms);
                        now.face.vectors.push(
                            VectorArtifact::new(
                                pete_now::FACE_VECTOR_COLLECTION,
                                format!("face-alex-{t_ms}"),
                                vec![0.1, 0.2],
                            )
                            .with_model("test-face-model"),
                        );
                        now
                    };

                    let first = runtime
                        .tick(person_now(100), ExperienceLatent::default(), Vec::new())
                        .await
                        .unwrap();
                    let request = first
                        .skill_request
                        .clone()
                        .expect("recognized encounter should propose greeting skill");
                    assert_eq!(request.goal_id, Some(GoalId::new("greet_person")));
                    assert_eq!(request.skill_id, SkillId::RuntimeLoaded);
                    assert_eq!(
                        request.implementation_id.as_deref(),
                        Some("motherbrain.greet")
                    );
                    assert!(!first
                        .frame
                        .now
                        .extensions
                        .get("event_scripts")
                        .is_some_and(|scripts| scripts.get("face-detected").is_some()));

                    let mut skills = PossessorSkillRuntime::default();
                    let mut cockpit = SimCockpit::new().with_unscoped_bench_mode();
                    let events = cockpit.get_events_since(0).unwrap();
                    let status_summary = cockpit.get_status().unwrap().summary();
                    let mut skill_now = first.frame.now.clone();
                    let (started, _) = skills.step(
                        &mut cockpit,
                        &request,
                        &skill_now,
                        &status_summary,
                        false,
                        &events,
                        100,
                    );
                    assert_ne!(started.phase, SkillPhase::Terminal);
                    let mut completed = started;
                    for now_ms in [200, 300, 400] {
                        skill_now.t_ms = now_ms;
                        skill_now.body.last_update_ms = now_ms;
                        (completed, _) = skills.step(
                            &mut cockpit,
                            &request,
                            &skill_now,
                            &status_summary,
                            false,
                            &events,
                            now_ms,
                        );
                        if completed.phase == SkillPhase::Terminal {
                            break;
                        }
                    }
                    assert_eq!(completed.outcome, Some(SkillOutcome::Completed));
                    assert_eq!(completed.progress, Some(1.0));
                    let execution = skills.provenance.as_ref().unwrap();
                    assert!(execution
                        .get("observations")
                        .and_then(serde_json::Value::as_array)
                        .is_some_and(|observations| observations.iter().any(|observation| {
                            observation.get("kind").and_then(serde_json::Value::as_str)
                                == Some("social_acknowledgment")
                        })));
                    assert!(execution
                        .get("trace")
                        .and_then(serde_json::Value::as_array)
                        .is_some_and(|trace| trace.iter().any(|event| {
                            event.get("kind").and_then(serde_json::Value::as_str)
                                == Some("primitive")
                                && event.get("operation").and_then(serde_json::Value::as_str)
                                    == Some("say")
                                && event
                                    .pointer("/detail/text")
                                    .and_then(serde_json::Value::as_str)
                                    == Some("Hello Alex.")
                        })));

                    let mut next_now = person_now(300);
                    skills.annotate_now(&mut next_now);
                    runtime.observe_skill_status(&completed);
                    let next = runtime
                        .tick(next_now, ExperienceLatent::default(), Vec::new())
                        .await
                        .unwrap();
                    let interaction = next
                        .frame
                        .now
                        .world
                        .social
                        .active_interaction
                        .as_ref()
                        .unwrap();
                    assert!(interaction.has_acknowledgment(
                        &pete_now::PersonId("person:alex".to_string()),
                        pete_now::SocialAcknowledgmentKind::GreetingAttempted,
                    ));
                    assert_ne!(
                        next.frame.now.extensions["goal_system"]["selection"]["selected_goal"]
                            .as_str(),
                        Some("greet_person")
                    );
                    assert!(next.skill_request.as_ref().is_none_or(|request| {
                        request.goal_id != Some(GoalId::new("greet_person"))
                    }));
                    let mut following_now = person_now(400);
                    skills.annotate_now(&mut following_now);
                    let following = runtime
                        .tick(following_now, ExperienceLatent::default(), Vec::new())
                        .await
                        .unwrap();
                    assert_ne!(
                        following.frame.now.world.self_model.active_goal.as_deref(),
                        Some("greet_person")
                    );
                });
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn dock_alignment_skill_follows_ir_and_fails_stopped_when_the_gradient_disappears() {
    let request = SkillRequest {
        skill_id: SkillId::AlignWithDock,
        maximum_duration_ms: 2_000,
        ..SkillRequest::default()
    };
    let mut body = BodySense {
        infrared_character: 254,
        ..BodySense::default()
    };
    let mut cockpit = SimCockpit::new().with_unscoped_bench_mode();
    let events = cockpit.get_events_since(0).unwrap();
    let mut runtime = PossessorSkillRuntime::default();

    let _ = possessor_test_step(
        &mut runtime,
        &mut cockpit,
        &request,
        &body,
        false,
        &events,
        1,
    );
    let (running, command_sent) = possessor_test_step(
        &mut runtime,
        &mut cockpit,
        &request,
        &body,
        false,
        &events,
        101,
    );
    assert!(command_sent);
    assert_eq!(running.phase, SkillPhase::Running);

    body.infrared_character = 0;
    let (lost, stop_sent) = possessor_test_step(
        &mut runtime,
        &mut cockpit,
        &request,
        &body,
        false,
        &events,
        201,
    );
    assert!(stop_sent);
    assert_eq!(lost.phase, SkillPhase::Terminal);
    assert_eq!(lost.outcome, Some(SkillOutcome::Failed));
    assert!(lost.reason.unwrap().contains("IR gradient"));
}

#[test]
fn charging_or_home_base_contact_completes_dock_alignment_operation() {
    let request = SkillRequest {
        skill_id: SkillId::AlignWithDock,
        ..SkillRequest::default()
    };
    for (body_charging, home_base_contact) in [(true, false), (false, true)] {
        let mut cockpit = SimCockpit::new().with_unscoped_bench_mode();
        let events = cockpit.get_events_since(0).unwrap();
        let mut state = EmbodiedLuaDriverState::default();
        let status = StatusSummary::from_raw("");
        let mut driver = RealLuaOrganDriver {
            cockpit: &mut cockpit,
            request: &request,
            status: &status,
            home_base_contact,
            state: &mut state,
            command_sent: false,
        };
        let now = Now::blank(
            100,
            BodySense {
                charging: body_charging,
                ..BodySense::default()
            },
        );
        let result = driver.poll(
            &HostOperation::AlignWithDock,
            OperationContext {
                operation_id: 1,
                child_id: 0,
                first_poll: true,
                elapsed_ms: 0,
                now_ms: 100,
                primitive_ttl_ms: 250,
            },
            &now,
            &events,
        );

        assert!(matches!(result, OrganPoll::Completed(_)));
        assert!(driver.command_sent);
    }
}

#[test]
fn home_base_contact_does_not_complete_dock_alignment_skill_without_charging() {
    let request = SkillRequest {
        skill_id: SkillId::AlignWithDock,
        maximum_duration_ms: 2_000,
        ..SkillRequest::default()
    };
    let mut cockpit = SimCockpit::new().with_unscoped_bench_mode();
    let events = cockpit.get_events_since(0).unwrap();
    let mut runtime = PossessorSkillRuntime::default();
    let mut status = SkillStatus::default();

    for now_ms in (1..=1_301).step_by(100) {
        (status, _) = possessor_test_step(
            &mut runtime,
            &mut cockpit,
            &request,
            &BodySense::default(),
            true,
            &events,
            now_ms,
        );
        if status.phase == SkillPhase::Terminal {
            break;
        }
    }

    assert_eq!(status.phase, SkillPhase::Terminal);
    assert_eq!(status.outcome, Some(SkillOutcome::PostconditionFailed));
    assert!(status
        .reason
        .as_deref()
        .is_some_and(|reason| reason.contains("fresh Create OI charging telemetry")));
}

fn poll_verify_charging(
    status_raw: &str,
    body_charging: bool,
    home_base_contact: bool,
    elapsed_ms: u64,
) -> OrganPoll {
    let mut cockpit = SimCockpit::new().with_unscoped_bench_mode();
    let events = cockpit.get_events_since(0).unwrap();
    let request = SkillRequest {
        skill_id: SkillId::ReturnToDock,
        ..SkillRequest::default()
    };
    let status = StatusSummary::from_raw(status_raw);
    let mut state = EmbodiedLuaDriverState::default();
    let mut driver = RealLuaOrganDriver {
        cockpit: &mut cockpit,
        request: &request,
        status: &status,
        home_base_contact,
        state: &mut state,
        command_sent: false,
    };
    let now = Now::blank(
        1_000,
        BodySense {
            charging: body_charging,
            ..BodySense::default()
        },
    );

    driver.poll(
        &HostOperation::VerifyCharging,
        OperationContext {
            operation_id: 1,
            child_id: 0,
            first_poll: elapsed_ms == 0,
            elapsed_ms,
            now_ms: 1_000,
            primitive_ttl_ms: 250,
        },
        &now,
        &events,
    )
}

#[test]
fn verify_charging_requires_fresh_active_create_oi_state() {
    let fresh_not_charging = r#"{
        "uptime_ms": 1000,
        "create_sensors": {
            "complete_packet_count": 1,
            "last_complete_packet_timestamp_ms": 1000,
            "charging_state": 0
        }
    }"#;

    for (body_charging, home_base_contact) in [(false, true), (true, false)] {
        assert!(matches!(
            poll_verify_charging(
                fresh_not_charging,
                body_charging,
                home_base_contact,
                1_000,
            ),
            OrganPoll::Failed(SkillFailure {
                outcome: SkillOutcome::PostconditionFailed,
                ..
            })
        ));
    }

    for charging_state in [4, 5] {
        let waiting_or_fault = format!(
            r#"{{
                "uptime_ms": 1000,
                "create_sensors": {{
                    "complete_packet_count": 1,
                    "last_complete_packet_timestamp_ms": 1000,
                    "charging_state": {charging_state}
                }}
            }}"#
        );
        assert!(matches!(
            poll_verify_charging(&waiting_or_fault, true, true, 1_000),
            OrganPoll::Failed(SkillFailure {
                outcome: SkillOutcome::PostconditionFailed,
                ..
            })
        ));
    }

    let stale_charging = r#"{
        "uptime_ms": 1000,
        "create_sensors": {
            "complete_packet_count": 1,
            "last_complete_packet_timestamp_ms": 499,
            "charging_state": 2
        }
    }"#;
    assert!(matches!(
        poll_verify_charging(stale_charging, true, true, 1_000),
        OrganPoll::Failed(SkillFailure {
            outcome: SkillOutcome::PostconditionFailed,
            ..
        })
    ));

    let fresh_charging = r#"{
        "uptime_ms": 1000,
        "create_sensors": {
            "complete_packet_count": 2,
            "last_complete_packet_timestamp_ms": 750,
            "charging_state": 2
        }
    }"#;
    let OrganPoll::Completed(result) =
        poll_verify_charging(fresh_charging, false, false, 0)
    else {
        panic!("fresh Create charging state should verify charging");
    };
    assert_eq!(result.get("charging"), Some(&json!(true)));
    assert_eq!(result.get("source"), Some(&json!("create_oi")));
    assert_eq!(result.get("charging_state"), Some(&json!(2)));
    assert_eq!(result.get("body_packet_age_ms"), Some(&json!(250)));
}

#[test]
fn create_ir_adds_directional_charger_evidence_and_bias_scores() {
    let mut now = Now::blank(
        100,
        BodySense {
            battery_level: 0.15,
            infrared_character: 246,
            ..BodySense::default()
        },
    );

    apply_create_ir_charger_cue(&mut now);
    let observation = now.objects.observations.last().unwrap();
    assert_eq!(observation.class, ObjectClass::Charger);
    assert_eq!(observation.source, ObjectObservationSource::CreateIr);
    assert_eq!(observation.bearing_rad, -0.35);
    assert_eq!(observation.distance_m, None);
    assert_eq!(charger_signal_scores(&now), (0.55, 0.85));

    let mut input = test_conductor_input(ActionPrimitive::Stop);
    input.body = now.body.clone();
    input.charger_near_score = charger_signal_scores(&now).0;
    input.charger_visible_score = charger_signal_scores(&now).1;
    assert_eq!(
        SimpleConductor::default().choose(input).unwrap(),
        ActionPrimitive::Approach {
            target: ApproachTarget::Charger,
        }
    );
}

#[test]
fn brainstem_contact_reflex_safety_preempts_a_possessor_skill() {
    let mut cockpit = SimCockpit::new().with_unscoped_bench_mode();
    let initial_events = cockpit.get_events_since(0).unwrap();
    let mut runtime = PossessorSkillRuntime::default();
    let request = SkillRequest {
        skill_id: SkillId::ApproachTarget,
        goal_id: None,
        behavior_id: None,
        target: Some(pete_now::EntityId("charger:17".to_string())),
        bearing_rad: Some(0.0),
        range_m: Some(2.0),
        stop_range_m: Some(0.3),
        maximum_duration_ms: 2_000,
        expected_progress: 0.9,
        ..SkillRequest::default()
    };
    let body = BodySense::default();
    let _ = possessor_test_step(
        &mut runtime,
        &mut cockpit,
        &request,
        &body,
        false,
        &initial_events,
        1,
    );
    let (_, command_sent) = possessor_test_step(
        &mut runtime,
        &mut cockpit,
        &request,
        &body,
        false,
        &initial_events,
        51,
    );
    assert!(command_sent);

    let cursor = initial_events.next_seq.saturating_sub(1);
    cockpit.set_bump(true, false);
    let reflex_events = cockpit.get_events_since(cursor).unwrap();
    let (preempted, command_sent) = possessor_test_step(
        &mut runtime,
        &mut cockpit,
        &request,
        &body,
        false,
        &reflex_events,
        100,
    );

    assert!(command_sent, "preemption must send a safe stop");
    assert_eq!(preempted.phase, SkillPhase::Terminal);
    assert_eq!(preempted.outcome, Some(SkillOutcome::SafetyPreempted));
}
