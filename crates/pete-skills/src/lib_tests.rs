use super::*;
use pete_body::BodySense;
use pete_cockpit::{CockpitEvent, EventBatch};
use pete_conductor::{GoalId, SkillScriptStatus};
use pete_now::EntityId;
use std::collections::BTreeSet;
use std::fs;
use tempfile::TempDir;

#[derive(Clone, Debug)]
struct OperationRecord {
    operation_id: u64,
    child_id: u64,
    name: String,
    resource: Option<BodyResource>,
    started_at_ms: u64,
    ended_at_ms: Option<u64>,
}

#[derive(Default)]
struct FakeDriver {
    records: Vec<OperationRecord>,
    stops: Vec<(BodyResource, String)>,
    fail_operations: HashMap<String, SkillFailure>,
    panic_operations: HashSet<String>,
    duration_ms: HashMap<String, u64>,
}

impl FakeDriver {
    fn duration_for(&self, operation: &HostOperation) -> u64 {
        self.duration_ms
            .get(operation.name())
            .copied()
            .unwrap_or(200)
    }
}

impl OrganDriver for FakeDriver {
    fn poll(
        &mut self,
        operation: &HostOperation,
        context: OperationContext,
        _now: &Now,
        _events: &EventBatch,
    ) -> OrganPoll {
        if context.first_poll {
            self.records.push(OperationRecord {
                operation_id: context.operation_id,
                child_id: context.child_id,
                name: operation.name().to_string(),
                resource: operation.resource(),
                started_at_ms: context.now_ms,
                ended_at_ms: None,
            });
        }
        assert!(
            !self.panic_operations.contains(operation.name()),
            "simulated organ driver panic"
        );
        if let Some(failure) = self.fail_operations.get(operation.name()).cloned() {
            self.records
                .iter_mut()
                .find(|record| record.operation_id == context.operation_id)
                .unwrap()
                .ended_at_ms = Some(context.now_ms);
            return OrganPoll::Failed(failure.for_operation(operation));
        }
        if context.elapsed_ms >= self.duration_for(operation) {
            self.records
                .iter_mut()
                .find(|record| record.operation_id == context.operation_id)
                .unwrap()
                .ended_at_ms = Some(context.now_ms);
            return OrganPoll::Completed(json!({
                "operation": operation.name(),
                "child_id": context.child_id,
            }));
        }
        OrganPoll::Pending {
            progress: operation.resource().map(|_| {
                (
                    "goal_progress".to_string(),
                    context.elapsed_ms as f32 / 200.0,
                )
            }),
            primitive: operation.resource().map(|resource| PrimitiveIntent {
                operation_id: context.operation_id,
                child_id: context.child_id,
                operation: operation.name().to_string(),
                resource: Some(resource),
                emitted_at_ms: context.now_ms,
                detail: json!({"fake": true}),
            }),
        }
    }

    fn stop(&mut self, resource: BodyResource, reason: &SkillFailure) {
        self.stops.push((resource, reason.kind.clone()));
    }
}

fn empty_events() -> EventBatch {
    EventBatch {
        since_seq: 0,
        oldest_seq: 0,
        next_seq: 0,
        dropped_before_seq: 0,
        events: Vec::new(),
    }
}

fn event(kind: CockpitEventKind) -> EventBatch {
    EventBatch {
        events: vec![CockpitEvent {
            seq: 1,
            kind,
            a: 0,
            b: 0,
            c: 0,
        }],
        next_seq: 2,
        ..empty_events()
    }
}

fn request(skill_id: SkillId) -> SkillRequest {
    SkillRequest {
        skill_id,
        goal_id: Some(GoalId::new("test_goal")),
        target: Some(EntityId("food:apple".to_string())),
        bearing_rad: Some(0.5),
        range_m: Some(2.0),
        stop_range_m: Some(0.2),
        maximum_duration_ms: 10_000,
        progress_metric: "goal_progress".to_string(),
        progress_baseline: Some(0.0),
        ..SkillRequest::default()
    }
}

fn write_skill(directory: &Path, function_name: &str, source: &str) {
    fs::write(directory.join(format!("{function_name}.lua")), source).unwrap();
}

fn runtime_with(function_name: &str, source: &str) -> (TempDir, LuaSkillRuntime, Now) {
    let directory = TempDir::new().unwrap();
    write_skill(directory.path(), function_name, source);
    let config = LuaSkillConfig {
        directory: directory.path().to_path_buf(),
        namespace: "test".to_string(),
        instruction_budget: 100_000,
        activation_budget: Duration::from_millis(50),
        memory_limit_bytes: 2 * 1024 * 1024,
        maximum_result_bytes: 16 * 1024,
        maximum_operation_ms: 2_000,
    };
    let runtime = LuaSkillRuntime::load(config).unwrap();
    let mut now = Now::blank(0, BodySense::default());
    now.objects.observations.push(pete_now::ObjectObservation {
        label: "apple".to_string(),
        class: ObjectClass::Unknown,
        bearing_rad: 0.5,
        distance_m: Some(2.0),
        confidence: 1.0,
        source: pete_now::ObjectObservationSource::Sim,
    });
    (directory, runtime, now)
}

fn advance(
    runtime: &mut LuaSkillRuntime,
    driver: &mut FakeDriver,
    now: &mut Now,
    ticks: usize,
) -> SkillStatus {
    let mut status = SkillStatus::default();
    for _ in 0..ticks {
        status = runtime
            .step(now, &empty_events(), driver)
            .expect("foreground invocation");
        if status.phase == SkillPhase::Terminal {
            break;
        }
        now.t_ms += 100;
        now.body.last_update_ms = now.t_ms;
    }
    status
}

fn completed_result(runtime: &LuaSkillRuntime) -> Option<Value> {
    runtime.trace().iter().rev().find_map(|event| match event {
        SkillTraceEvent::Completed { result, .. } => result.clone(),
        _ => None,
    })
}

#[test]
fn valid_skills_load_with_hash_path_and_runtime_version() {
    let (_directory, runtime, _now) = runtime_with(
        "stopAndStabilize",
        "function stopAndStabilize(args) return 'ok' end",
    );
    let skills = runtime.discoverable_skills();
    assert_eq!(skills.len(), 1);
    assert_eq!(skills[0].skill_id, "test.stopAndStabilize");
    assert_eq!(skills[0].source_hash.len(), 64);
    assert!(skills[0].source_path.ends_with("stopAndStabilize.lua"));
    assert!(skills[0].runtime_version.contains("Lua 5.4"));
}

#[test]
fn newly_discovered_skill_can_run_by_inferred_runtime_id() {
    let (_directory, mut runtime, mut now) = runtime_with(
        "waveHello",
        "function waveHello(args) return args.implementation_id end",
    );
    let request = SkillRequest::runtime_loaded("test.waveHello");
    runtime.start(request, &now).unwrap();
    let status = advance(&mut runtime, &mut FakeDriver::default(), &mut now, 2);
    assert_eq!(status.outcome, Some(SkillOutcome::Completed));
    assert_eq!(completed_result(&runtime), Some(json!("test.waveHello")));
}

#[test]
fn invalid_reload_leaves_prior_generation_active() {
    let (directory, mut runtime, _now) = runtime_with(
        "stopAndStabilize",
        "function stopAndStabilize(args) return 'valid' end",
    );
    let generation = runtime.generation_hash().to_string();
    write_skill(
        directory.path(),
        "stopAndStabilize",
        "function stopAndStabilize(",
    );
    assert!(runtime.reload().is_err());
    assert_eq!(runtime.generation_hash(), generation);
    assert!(runtime.last_reload_error().is_some());
}

#[test]
fn sandbox_removes_host_access_and_raw_cockpit() {
    let (_directory, mut runtime, mut now) = runtime_with(
        "stopAndStabilize",
        r#"
                function stopAndStabilize(args)
                    assert(io == nil and os == nil and debug == nil)
                    assert(package == nil and dofile == nil and loadfile == nil and load == nil)
                    assert(coroutine == nil and rawget == nil and rawset == nil)
                    assert(Cockpit == nil and cockpit == nil and socket == nil)
                    assert(math.random == nil and math.randomseed == nil)
                    return "sandboxed"
                end
            "#,
    );
    runtime
        .start(request(SkillId::StopAndStabilize), &now)
        .unwrap();
    let status = advance(&mut runtime, &mut FakeDriver::default(), &mut now, 2);
    assert_eq!(status.outcome, Some(SkillOutcome::Completed));
    assert_eq!(completed_result(&runtime), Some(json!("sandboxed")));
}

#[test]
fn active_invocation_is_pinned_across_atomic_hot_reload() {
    let (directory, mut runtime, mut now) = runtime_with(
        "stopAndStabilize",
        "function stopAndStabilize(args) drive(0.05, 400); return 'old' end",
    );
    runtime
        .start(request(SkillId::StopAndStabilize), &now)
        .unwrap();
    let old_hash = runtime.diagnostics().source_hash.unwrap();
    let _ = advance(&mut runtime, &mut FakeDriver::default(), &mut now, 1);
    write_skill(
        directory.path(),
        "stopAndStabilize",
        "function stopAndStabilize(args) return 'new' end",
    );
    assert!(runtime.reload().unwrap());
    assert_eq!(
        runtime.diagnostics().source_hash.as_deref(),
        Some(old_hash.as_str())
    );
    let mut driver = FakeDriver::default();
    let status = advance(&mut runtime, &mut driver, &mut now, 10);
    assert_eq!(status.outcome, Some(SkillOutcome::Completed));
    assert_eq!(completed_result(&runtime), Some(json!("old")));
    runtime.take_terminal();

    now.t_ms += 100;
    runtime
        .start(request(SkillId::StopAndStabilize), &now)
        .unwrap();
    let status = advance(&mut runtime, &mut driver, &mut now, 2);
    assert_eq!(status.outcome, Some(SkillOutcome::Completed));
    assert_eq!(completed_result(&runtime), Some(json!("new")));
    assert_ne!(
        runtime.diagnostics().source_hash.as_deref(),
        Some(old_hash.as_str())
    );
}

#[test]
fn infinite_loop_exhausts_budget_without_dispatching() {
    let (_directory, mut runtime, mut now) = runtime_with(
        "stopAndStabilize",
        "function stopAndStabilize(args) while true do end end",
    );
    runtime
        .start(request(SkillId::StopAndStabilize), &now)
        .unwrap();
    let mut driver = FakeDriver::default();
    let status = advance(&mut runtime, &mut driver, &mut now, 2);
    assert_eq!(status.outcome, Some(SkillOutcome::BudgetExceeded));
    assert!(driver.records.is_empty());
    assert!(runtime.diagnostics().held_resources.is_empty());
}

#[test]
fn child_budget_exhaustion_stops_active_sibling_command() {
    let (_directory, mut runtime, mut now) = runtime_with(
        "approachTarget",
        r#"
                function approachTarget(args)
                    together(
                        function() drive(0.05, 1000) end,
                        function()
                            scan()
                            while true do end
                        end
                    )
                end
            "#,
    );
    runtime
        .start(request(SkillId::ApproachTarget), &now)
        .unwrap();
    let mut driver = FakeDriver::default();
    driver.duration_ms.insert("drive".to_string(), 1_000);
    driver.duration_ms.insert("scan".to_string(), 100);
    let status = advance(&mut runtime, &mut driver, &mut now, 8);
    assert_eq!(status.outcome, Some(SkillOutcome::BudgetExceeded));
    assert!(driver
        .stops
        .iter()
        .any(|(resource, _)| *resource == BodyResource::Locomotion));
    assert!(runtime.diagnostics().held_resources.is_empty());
}

#[test]
fn excessive_recursion_fails_safely() {
    let (_directory, mut runtime, mut now) = runtime_with(
        "stopAndStabilize",
        r#"
                local function recurse(n)
                    return n + recurse(n + 1)
                end
                function stopAndStabilize(args)
                    return recurse(0)
                end
            "#,
    );
    runtime
        .start(request(SkillId::StopAndStabilize), &now)
        .unwrap();
    let status = advance(&mut runtime, &mut FakeDriver::default(), &mut now, 2);
    assert!(matches!(
        status.outcome,
        Some(SkillOutcome::BudgetExceeded | SkillOutcome::ScriptError)
    ));
    assert!(runtime.diagnostics().held_resources.is_empty());
}

#[test]
fn excessive_memory_fails_safely() {
    let (_directory, mut runtime, mut now) = runtime_with(
        "stopAndStabilize",
        r#"
                function stopAndStabilize(args)
                    return string.rep("x", 3 * 1024 * 1024)
                end
            "#,
    );
    runtime
        .start(request(SkillId::StopAndStabilize), &now)
        .unwrap();
    let status = advance(&mut runtime, &mut FakeDriver::default(), &mut now, 2);
    assert_eq!(status.outcome, Some(SkillOutcome::BudgetExceeded));
    assert!(runtime.diagnostics().held_resources.is_empty());
}

#[test]
fn plain_nested_functions_suspend_and_resume_with_return_values() {
    let (_directory, mut runtime, mut now) = runtime_with(
        "stopAndStabilize",
        r#"
                local function inner()
                    drive(0.05, 200)
                    return 41
                end
                function stopAndStabilize(args)
                    return inner() + 1
                end
            "#,
    );
    runtime
        .start(request(SkillId::StopAndStabilize), &now)
        .unwrap();
    let mut driver = FakeDriver::default();
    let first = advance(&mut runtime, &mut driver, &mut now, 1);
    assert_eq!(first.phase, SkillPhase::Running);
    assert!(driver.records.is_empty());
    let status = advance(&mut runtime, &mut driver, &mut now, 8);
    assert_eq!(status.outcome, Some(SkillOutcome::Completed));
    assert_eq!(completed_result(&runtime), Some(json!(42)));
    assert_eq!(driver.records.len(), 1);
    assert!(runtime.diagnostics().held_resources.is_empty());
}

#[test]
fn typed_failure_can_be_handled_with_try() {
    let (_directory, mut runtime, mut now) = runtime_with(
        "approachTarget",
        r#"
                function approachTarget(args)
                    local ok, result = try(function()
                        approach(args.target, 0.2)
                    end)
                    if not ok and result.kind == "contact_withdrawal" then
                        return "handled"
                    end
                    if not ok then error(result) end
                    return "unexpected"
                end
            "#,
    );
    runtime
        .start(request(SkillId::ApproachTarget), &now)
        .unwrap();
    let mut driver = FakeDriver::default();
    driver.fail_operations.insert(
        "approach".to_string(),
        SkillFailure::new(
            SkillOutcome::SafetyPreempted,
            "contact_withdrawal",
            "contact reflex",
        ),
    );
    let status = advance(&mut runtime, &mut driver, &mut now, 6);
    assert_eq!(status.outcome, Some(SkillOutcome::Completed));
    assert_eq!(completed_result(&runtime), Some(json!("handled")));
    assert!(runtime.diagnostics().held_resources.is_empty());
}

#[test]
fn typed_failure_rethrown_from_lua_preserves_original_outcome() {
    let (_directory, mut runtime, mut now) = runtime_with(
        "stopAndStabilize",
        r#"
                function stopAndStabilize(args)
                    local ok, result = try(function() scan() end)
                    if not ok then error(result) end
                end
            "#,
    );
    runtime
        .start(request(SkillId::StopAndStabilize), &now)
        .unwrap();
    let mut driver = FakeDriver::default();
    driver.fail_operations.insert(
        "scan".to_string(),
        SkillFailure::new(
            SkillOutcome::CapabilityUnavailable,
            "capability_unavailable",
            "gaze is absent",
        ),
    );
    let status = advance(&mut runtime, &mut driver, &mut now, 8);
    assert_eq!(status.outcome, Some(SkillOutcome::CapabilityUnavailable));
    assert_eq!(status.reason.as_deref(), Some("gaze is absent"));
}

#[test]
fn organ_driver_panic_stops_and_releases_owned_resource() {
    let (_directory, mut runtime, mut now) = runtime_with(
        "stopAndStabilize",
        "function stopAndStabilize(args) drive(0.05, 200) end",
    );
    runtime
        .start(request(SkillId::StopAndStabilize), &now)
        .unwrap();
    let mut driver = FakeDriver::default();
    driver.panic_operations.insert("drive".to_string());
    let status = advance(&mut runtime, &mut driver, &mut now, 4);
    assert_eq!(status.outcome, Some(SkillOutcome::ScriptError));
    assert_eq!(driver.stops.len(), 1);
    assert!(runtime.diagnostics().held_resources.is_empty());
}

#[test]
fn foreground_is_exclusive_and_explicit_cancellation_unwinds_resources() {
    let (_directory, mut runtime, mut now) = runtime_with(
        "stopAndStabilize",
        "function stopAndStabilize(args) drive(0.05, 1000) end",
    );
    runtime
        .start(request(SkillId::StopAndStabilize), &now)
        .unwrap();
    assert!(runtime
        .start(request(SkillId::StopAndStabilize), &now)
        .is_err());
    let mut driver = FakeDriver::default();
    let _ = advance(&mut runtime, &mut driver, &mut now, 2);
    let status = runtime
        .cancel(
            &mut driver,
            SkillOutcome::Cancelled,
            "operator_preempted",
            "operator selected a stronger intention",
            now.t_ms,
        )
        .unwrap();
    assert_eq!(status.outcome, Some(SkillOutcome::Cancelled));
    assert_eq!(driver.stops.len(), 1);
    assert!(runtime.diagnostics().held_resources.is_empty());
}

#[test]
fn together_overlaps_disjoint_organs_and_preserves_result_order() {
    let (_directory, mut runtime, mut now) = runtime_with(
        "approachTarget",
        r#"
                function approachTarget(args)
                    return together(
                        function() drive(0.05, 200); return "move" end,
                        function() scan(); return "look" end,
                        function() say("hello"); return "speak" end
                    )
                end
            "#,
    );
    runtime
        .start(request(SkillId::ApproachTarget), &now)
        .unwrap();
    let mut driver = FakeDriver::default();
    let status = advance(&mut runtime, &mut driver, &mut now, 10);
    assert_eq!(status.outcome, Some(SkillOutcome::Completed));
    assert_eq!(driver.records.len(), 3);
    let starts: BTreeSet<_> = driver
        .records
        .iter()
        .map(|record| record.started_at_ms)
        .collect();
    assert_eq!(starts.len(), 1, "disjoint organs must overlap");
    assert_eq!(
        driver
            .records
            .iter()
            .map(|record| record.resource)
            .collect::<BTreeSet<_>>(),
        BTreeSet::from([
            Some(BodyResource::Locomotion),
            Some(BodyResource::Gaze),
            Some(BodyResource::Voice),
        ])
    );
    assert_eq!(
        completed_result(&runtime),
        Some(json!([["move"], ["look"], ["speak"]]))
    );
}

#[test]
fn together_serializes_same_resource_in_child_order_without_busy_polling() {
    let (_directory, mut runtime, mut now) = runtime_with(
        "approachTarget",
        r#"
                function approachTarget(args)
                    return together(
                        function() drive(0.04, 200); return 1 end,
                        function() turnBy(0.5); return 2 end
                    )
                end
            "#,
    );
    runtime
        .start(request(SkillId::ApproachTarget), &now)
        .unwrap();
    let mut driver = FakeDriver::default();
    let _ = advance(&mut runtime, &mut driver, &mut now, 2);
    let diagnostics = runtime.diagnostics();
    assert_eq!(
        diagnostics.held_resources.get(&BodyResource::Locomotion),
        Some(&1)
    );
    assert_eq!(
        diagnostics.waiting_resources.get(&BodyResource::Locomotion),
        Some(&vec![2])
    );
    let status = advance(&mut runtime, &mut driver, &mut now, 10);
    assert_eq!(status.outcome, Some(SkillOutcome::Completed));
    assert_eq!(driver.records[0].child_id, 1);
    assert_eq!(driver.records[1].child_id, 2);
    assert!(
        driver.records[1].started_at_ms
            >= driver.records[0]
                .ended_at_ms
                .expect("first operation ended")
    );
}

#[test]
fn together_is_fail_fast_and_stops_siblings() {
    let (_directory, mut runtime, mut now) = runtime_with(
        "approachTarget",
        r#"
                function approachTarget(args)
                    together(
                        function() drive(0.04, 1000) end,
                        function() scan() end,
                        function() say("still speaking") end
                    )
                end
            "#,
    );
    runtime
        .start(request(SkillId::ApproachTarget), &now)
        .unwrap();
    let mut driver = FakeDriver::default();
    driver.fail_operations.insert(
        "scan".to_string(),
        SkillFailure::new(
            SkillOutcome::CapabilityUnavailable,
            "capability_unavailable",
            "gaze is absent",
        ),
    );
    let status = advance(&mut runtime, &mut driver, &mut now, 8);
    assert_eq!(status.outcome, Some(SkillOutcome::CapabilityUnavailable));
    assert!(driver
        .stops
        .iter()
        .any(|(resource, _)| *resource == BodyResource::Locomotion));
    assert!(driver
        .stops
        .iter()
        .any(|(resource, _)| *resource == BodyResource::Voice));
    assert!(runtime.diagnostics().held_resources.is_empty());
}

#[test]
fn parent_cancellation_and_reflex_preemption_cancel_together_children() {
    for preempt in [false, true] {
        let (_directory, mut runtime, mut now) = runtime_with(
            "approachTarget",
            r#"
                    function approachTarget(args)
                        together(
                            function() drive(0.04, 1000) end,
                            function() say("working") end
                        )
                    end
                "#,
        );
        runtime
            .start(request(SkillId::ApproachTarget), &now)
            .unwrap();
        let mut driver = FakeDriver::default();
        let _ = advance(&mut runtime, &mut driver, &mut now, 2);
        let status = if preempt {
            runtime
                .step(
                    &now,
                    &event(CockpitEventKind::ContactWithdrawalStarted),
                    &mut driver,
                )
                .unwrap()
        } else {
            runtime
                .cancel(
                    &mut driver,
                    SkillOutcome::Cancelled,
                    "parent_cancelled",
                    "parent cancelled",
                    now.t_ms,
                )
                .unwrap()
        };
        assert_eq!(
            status.outcome,
            Some(if preempt {
                SkillOutcome::SafetyPreempted
            } else {
                SkillOutcome::Cancelled
            })
        );
        assert_eq!(driver.stops.len(), 2);
        assert!(runtime.diagnostics().held_resources.is_empty());
    }
}

#[test]
fn authority_and_transport_loss_cancel_active_skill_and_careful_escape() {
    for (kind, expected) in [
        (
            CockpitEventKind::AuthorityChanged,
            SkillOutcome::AuthorityLost,
        ),
        (
            CockpitEventKind::HeartbeatExpired,
            SkillOutcome::TransportLost,
        ),
    ] {
        let (_directory, mut runtime, mut now) = runtime_with(
            "releasePersistentBumper",
            r#"
                    function releasePersistentBumper(args)
                        carefully("bumper_front", function() retreat(100) end)
                    end
                "#,
        );
        now.body.flags.bump_left = true;
        runtime
            .start(request(SkillId::ReleasePersistentBumper), &now)
            .unwrap();
        let mut driver = FakeDriver::default();
        let _ = advance(&mut runtime, &mut driver, &mut now, 2);
        let status = runtime.step(&now, &event(kind), &mut driver).unwrap();
        assert_eq!(status.outcome, Some(expected));
        assert!(driver
            .stops
            .iter()
            .any(|(resource, _)| *resource == BodyResource::Locomotion));
        assert!(runtime.diagnostics().held_resources.is_empty());
    }
}

#[test]
fn nested_together_is_deterministic_and_safe() {
    let (_directory, mut runtime, mut now) = runtime_with(
        "approachTarget",
        r#"
                function approachTarget(args)
                    return together(
                        function()
                            return together(
                                function() drive(0.04, 200); return "a" end,
                                function() say("nested"); return "b" end
                            )
                        end,
                        function() scan(); return "c" end
                    )
                end
            "#,
    );
    runtime
        .start(request(SkillId::ApproachTarget), &now)
        .unwrap();
    let status = advance(&mut runtime, &mut FakeDriver::default(), &mut now, 12);
    assert_eq!(status.outcome, Some(SkillOutcome::Completed));
    assert_eq!(
        completed_result(&runtime),
        Some(json!([[[["a"], ["b"]]], ["c"]]))
    );
}

#[test]
fn careful_allows_only_acknowledged_hazard_retreat() {
    let (_directory, mut runtime, mut now) = runtime_with(
        "stopAndStabilize",
        r#"
                function stopAndStabilize(args)
                    carefully("bumper_front", function()
                        retreat(100)
                    end)
                end
            "#,
    );
    now.body.flags.bump_left = true;
    runtime
        .start(request(SkillId::StopAndStabilize), &now)
        .unwrap();
    let mut driver = FakeDriver::default();
    let status = advance(&mut runtime, &mut driver, &mut now, 8);
    assert_eq!(status.outcome, Some(SkillOutcome::Completed));
    assert!(matches!(driver.records[0].name.as_str(), "retreat"));
}

#[test]
fn careful_cannot_expand_envelope_or_suppress_absolute_hazard() {
    let source = r#"
            function stopAndStabilize(args)
                carefully("bumper_front", function()
                    drive(0.12, 1000)
                end)
            end
        "#;
    let (_directory, mut runtime, mut now) = runtime_with("stopAndStabilize", source);
    now.body.flags.bump_left = true;
    runtime
        .start(request(SkillId::StopAndStabilize), &now)
        .unwrap();
    let status = advance(&mut runtime, &mut FakeDriver::default(), &mut now, 5);
    assert_eq!(status.outcome, Some(SkillOutcome::Failed));
    assert!(status.reason.unwrap().contains("bounded retreat"));

    let (_directory, mut runtime, mut now) = runtime_with(
        "stopAndStabilize",
        r#"
                function stopAndStabilize(args)
                    carefully("bumper_front", function() retreat(100) end)
                end
            "#,
    );
    now.body.flags.bump_left = true;
    now.body.flags.wheel_drop = true;
    runtime
        .start(request(SkillId::StopAndStabilize), &now)
        .unwrap();
    let status = advance(&mut runtime, &mut FakeDriver::default(), &mut now, 5);
    assert_eq!(status.outcome, Some(SkillOutcome::SafetyPreempted));
    assert!(status.reason.unwrap().contains("absolute"));
}

#[test]
fn progress_is_explicit_bounded_and_retains_originating_goal() {
    let (_directory, mut runtime, mut now) = runtime_with(
        "stopAndStabilize",
        r#"
                function stopAndStabilize(args)
                    reportProgress("goal_progress", 0.65)
                    drive(0.04, 200)
                    return true
                end
            "#,
    );
    let request = request(SkillId::StopAndStabilize);
    runtime.start(request.clone(), &now).unwrap();
    let mut driver = FakeDriver::default();
    let running = advance(&mut runtime, &mut driver, &mut now, 2);
    assert_eq!(running.request.goal_id, request.goal_id);
    assert_eq!(running.progress, Some(0.65));
    let terminal = advance(&mut runtime, &mut driver, &mut now, 6);
    assert_eq!(terminal.progress, Some(1.0));
    assert_eq!(
        terminal.script,
        Some(SkillScriptStatus {
            skill_id: "test.stopAndStabilize".to_string(),
            source_hash: runtime.discoverable_skills()[0].source_hash.clone(),
            source_path: runtime.discoverable_skills()[0]
                .source_path
                .display()
                .to_string(),
            current_function: Some("stopAndStabilize".to_string()),
            current_operation: None,
            held_resources: Vec::new(),
            waiting_resources: Vec::new(),
            active_children: 0,
        })
    );
}

#[test]
fn greet_uses_canonical_person_identity_and_records_encounter_acknowledgment() {
    let source = r#"
            function greet(args)
                local person = args.target
                require(person ~= nil, "person required")
                together(
                    function() face(person) end,
                    function() say("Hello " .. person.name .. ".") end
                )
                acknowledge(person)
                reportProgress("social_acknowledgment", 1.0)
                return {
                    person_id = person.id,
                    name = person.name,
                    acknowledged = true,
                }
            end
        "#;
    let (_directory, mut runtime, _) = runtime_with("greet", source);
    let mut raw = Now::blank(0, BodySense::default());
    raw.objects.observations.push(pete_now::ObjectObservation {
        label: "Alex".to_string(),
        class: ObjectClass::Person,
        bearing_rad: 0.25,
        distance_m: Some(0.8),
        confidence: 0.95,
        source: pete_now::ObjectObservationSource::Kinect,
    });
    let mut now = pete_now::WorldModelUpdater::default()
        .update(raw, pete_now::WorldModelUpdateContext::default());
    let encounter_id = now
        .world
        .social
        .active_interaction
        .as_ref()
        .unwrap()
        .interaction_id
        .0
        .clone();
    let mut request = SkillRequest::runtime_loaded("test.greet");
    request.goal_id = Some(GoalId::new("greet_person"));
    request.behavior_id = Some(format!("greet:person:alex:{encounter_id}"));
    request.target = Some(EntityId("person:alex".to_string()));
    request.bearing_rad = Some(0.25);
    request.progress_metric = "social_acknowledgment".to_string();
    runtime.start(request, &now).unwrap();

    let mut driver = FakeDriver::default();
    let status = advance(&mut runtime, &mut driver, &mut now, 8);
    assert_eq!(status.outcome, Some(SkillOutcome::Completed));
    assert_eq!(status.progress, Some(1.0));
    assert_eq!(
        completed_result(&runtime),
        Some(json!({
            "person_id": "person:alex",
            "name": "Alex",
            "acknowledged": true,
        }))
    );
    assert_eq!(
        driver
            .records
            .iter()
            .map(|record| record.name.as_str())
            .collect::<BTreeSet<_>>(),
        BTreeSet::from(["face_bearing", "say"])
    );
    let record = runtime.execution_record().unwrap();
    assert_eq!(record.execution_id, status.execution_id);
    assert_eq!(
        record.observations,
        vec![json!({
            "kind": "social_acknowledgment",
            "contract": "host_validated_social_acknowledgment_v1",
            "value": {
                "acknowledgment_id": format!(
                    "greet:{encounter_id}:person:alex:{}",
                    status.execution_id
                ),
                "interaction_id": encounter_id,
                "person_id": "person:alex",
                "occurred_at_ms": 200,
            },
            "provenance": "lua_skill",
        })]
    );
}
