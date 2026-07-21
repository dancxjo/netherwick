fn drain_new_requests(invocation: &mut Invocation, now_ms: u64) {
    let requests: Vec<_> = invocation
        .bridge
        .state
        .lock()
        .expect("skill bridge lock")
        .requests
        .drain(..)
        .collect();
    for request in requests {
        if let Some(resource) = request.operation.resource() {
            if invocation
                .active
                .values()
                .any(|active| active.request.child_id == request.child_id)
                || invocation.waiters.values().any(|waiters| {
                    waiters
                        .iter()
                        .any(|waiting| waiting.child_id == request.child_id)
                })
            {
                request.slot.finish(Err(SkillFailure::new(
                    SkillOutcome::Failed,
                    "multiple_exclusive_resources",
                    "a child cannot await a second exclusive organ",
                )
                .for_operation(&request.operation)));
                continue;
            }
            if invocation.owners.contains_key(&resource) {
                invocation
                    .waiters
                    .entry(resource)
                    .or_default()
                    .push_back(request.clone());
                push_invocation_trace(
                    invocation,
                    SkillTraceEvent::ResourceWaiting {
                        at_ms: now_ms,
                        operation_id: request.id,
                        child_id: request.child_id,
                        resource,
                    },
                );
            } else {
                acquire(invocation, request, resource, now_ms);
            }
        } else {
            invocation.active.insert(
                request.id,
                ActiveOperation {
                    request,
                    granted_at_ms: now_ms,
                    polls: 0,
                },
            );
        }
    }
}

fn acquire(
    invocation: &mut Invocation,
    request: OperationRequest,
    resource: BodyResource,
    now_ms: u64,
) {
    invocation.owners.insert(resource, request.id);
    push_invocation_trace(
        invocation,
        SkillTraceEvent::ResourceAcquired {
            at_ms: now_ms,
            operation_id: request.id,
            child_id: request.child_id,
            resource,
        },
    );
    invocation.active.insert(
        request.id,
        ActiveOperation {
            request,
            granted_at_ms: now_ms,
            polls: 0,
        },
    );
}

fn grant_waiting_operations(invocation: &mut Invocation, now_ms: u64) {
    let resources: Vec<_> = invocation.waiters.keys().copied().collect();
    for resource in resources {
        if invocation.owners.contains_key(&resource) {
            continue;
        }
        let request = invocation
            .waiters
            .get_mut(&resource)
            .and_then(VecDeque::pop_front);
        if let Some(request) = request {
            acquire(invocation, request, resource, now_ms);
        }
        if invocation
            .waiters
            .get(&resource)
            .is_some_and(VecDeque::is_empty)
        {
            invocation.waiters.remove(&resource);
        }
    }
}

fn expire_resource_waits(invocation: &mut Invocation, now_ms: u64) {
    for waiters in invocation.waiters.values_mut() {
        let mut retained = VecDeque::new();
        while let Some(request) = waiters.pop_front() {
            if now_ms.saturating_sub(request.requested_at_ms) >= request.timeout_ms {
                request.slot.finish(Err(SkillFailure::new(
                    SkillOutcome::TimedOut,
                    "resource_wait_timed_out",
                    format!(
                        "timed out waiting for {:?}",
                        request.operation.resource().unwrap_or_default()
                    ),
                )
                .for_operation(&request.operation)));
            } else {
                retained.push_back(request);
            }
        }
        *waiters = retained;
    }
    invocation.waiters.retain(|_, waiters| !waiters.is_empty());
}

fn service_active_operations<D: OrganDriver>(
    invocation: &mut Invocation,
    driver: &mut D,
    now: &Now,
    events: &EventBatch,
    maximum_operation_ms: u64,
) {
    let ids: Vec<_> = invocation.active.keys().copied().collect();
    let mut finished = Vec::new();
    for id in ids {
        let Some(active) = invocation.active.get_mut(&id) else {
            continue;
        };
        let elapsed_ms = now.t_ms.saturating_sub(active.granted_at_ms);
        let timeout_ms = active.request.timeout_ms.min(maximum_operation_ms);
        if elapsed_ms >= timeout_ms {
            finished.push((
                id,
                Err(SkillFailure::new(
                    SkillOutcome::TimedOut,
                    "operation_timed_out",
                    format!(
                        "{} exceeded {} ms",
                        active.request.operation.name(),
                        timeout_ms
                    ),
                )
                .for_operation(&active.request.operation)),
            ));
            continue;
        }
        let context = OperationContext {
            operation_id: id,
            child_id: active.request.child_id,
            first_poll: active.polls == 0,
            elapsed_ms,
            now_ms: now.t_ms,
            primitive_ttl_ms: PRIMITIVE_TTL_MS,
        };
        active.polls = active.polls.saturating_add(1);
        let polled = catch_unwind(AssertUnwindSafe(|| {
            driver.poll(&active.request.operation, context, now, events)
        }))
        .unwrap_or_else(|_| {
            OrganPoll::Failed(
                SkillFailure::new(
                    SkillOutcome::ScriptError,
                    "organ_driver_panic",
                    "bodily operation driver panicked",
                )
                .for_operation(&active.request.operation),
            )
        });
        match polled {
            OrganPoll::Pending {
                progress,
                primitive,
            } => {
                if let Some((key, value)) = progress {
                    invocation.diagnostics.progress.insert(key.clone(), value);
                    push_invocation_trace(
                        invocation,
                        SkillTraceEvent::Progress {
                            at_ms: now.t_ms,
                            key,
                            value,
                        },
                    );
                }
                if let Some(primitive) = primitive {
                    invocation.status.dispatch_count =
                        invocation.status.dispatch_count.saturating_add(1);
                    push_invocation_trace(invocation, SkillTraceEvent::Primitive(primitive));
                }
            }
            OrganPoll::Completed(value) => finished.push((id, Ok(value))),
            OrganPoll::Failed(failure) => finished.push((id, Err(failure))),
        }
    }
    for (id, result) in finished {
        finish_operation(invocation, driver, id, result, now.t_ms);
    }
}

fn finish_operation<D: OrganDriver>(
    invocation: &mut Invocation,
    driver: &mut D,
    id: u64,
    result: std::result::Result<Value, SkillFailure>,
    now_ms: u64,
) {
    let Some(active) = invocation.active.remove(&id) else {
        return;
    };
    if let Some(resource) = active.request.operation.resource() {
        invocation.owners.remove(&resource);
        if result.is_err() {
            driver.stop(
                resource,
                result.as_ref().err().expect("failed operation has error"),
            );
        }
        push_invocation_trace(
            invocation,
            SkillTraceEvent::ResourceReleased {
                at_ms: now_ms,
                operation_id: id,
                child_id: active.request.child_id,
                resource,
                reason: result
                    .as_ref()
                    .err()
                    .map_or_else(|| "completed".to_string(), |failure| failure.kind.clone()),
            },
        );
    }
    if let Err(failure) = &result {
        push_invocation_trace(
            invocation,
            SkillTraceEvent::Preempted {
                at_ms: now_ms,
                operation_id: id,
                failure: failure.clone(),
            },
        );
        invocation.diagnostics.last_preemption = Some(failure.clone());
    }
    active.request.slot.finish(result);
}

fn service_child_cancellations<D: OrganDriver>(
    invocation: &mut Invocation,
    driver: &mut D,
    now_ms: u64,
) {
    let cancellations: Vec<_> = invocation
        .bridge
        .state
        .lock()
        .expect("skill bridge lock")
        .cancelled_children
        .drain(..)
        .collect();
    for (children, failure) in cancellations {
        let active_ids: Vec<_> = invocation
            .active
            .iter()
            .filter_map(|(id, active)| children.contains(&active.request.child_id).then_some(*id))
            .collect();
        for id in active_ids {
            finish_operation(invocation, driver, id, Err(failure.clone()), now_ms);
        }
        for waiters in invocation.waiters.values_mut() {
            let mut retained = VecDeque::new();
            while let Some(request) = waiters.pop_front() {
                if children.contains(&request.child_id) {
                    request.slot.finish(Err(failure.clone()));
                } else {
                    retained.push_back(request);
                }
            }
            *waiters = retained;
        }
    }
}

fn cancel_invocation<D: OrganDriver>(
    invocation: &mut Invocation,
    driver: &mut D,
    failure: SkillFailure,
) {
    if invocation.result.is_some() {
        return;
    }
    invocation.bridge.cancelled.store(true, Ordering::Release);
    let ids: Vec<_> = invocation.active.keys().copied().collect();
    for id in ids {
        finish_operation(
            invocation,
            driver,
            id,
            Err(failure.clone()),
            invocation.status.updated_at_ms,
        );
    }
    for waiters in invocation.waiters.values_mut() {
        for request in waiters.drain(..) {
            request.slot.finish(Err(failure.clone()));
        }
    }
    invocation.waiters.clear();
    for request in invocation
        .bridge
        .state
        .lock()
        .expect("skill bridge lock")
        .requests
        .drain(..)
    {
        request.slot.finish(Err(failure.clone()));
    }
    invocation.result = Some(Err(failure));
    update_diagnostics(invocation);
}

fn update_diagnostics(invocation: &mut Invocation) {
    invocation.diagnostics.held_resources = invocation
        .owners
        .iter()
        .filter_map(|(resource, operation_id)| {
            invocation
                .active
                .get(operation_id)
                .map(|active| (*resource, active.request.child_id))
        })
        .collect();
    invocation.diagnostics.waiting_resources = invocation
        .waiters
        .iter()
        .map(|(resource, waiters)| {
            (
                *resource,
                waiters.iter().map(|request| request.child_id).collect(),
            )
        })
        .collect();
    invocation.diagnostics.current_operation = invocation
        .active
        .values()
        .min_by_key(|active| active.request.id)
        .map(|active| active.request.operation.name().to_string());
    let bridge = invocation.bridge.state.lock().expect("skill bridge lock");
    invocation.diagnostics.current_lua_function = bridge
        .current_functions
        .get(&0)
        .cloned()
        .or_else(|| Some(invocation.metadata.function_name.clone()));
    for (key, value) in &bridge.progress {
        invocation.diagnostics.progress.insert(key.clone(), *value);
    }
    for (key, value) in &bridge.reported_progress {
        invocation.diagnostics.progress.insert(key.clone(), *value);
    }
    invocation.diagnostics.active_together_children = bridge
        .child_parent
        .iter()
        .map(|(child_id, (parent, order))| {
            let active = invocation
                .active
                .values()
                .find(|active| active.request.child_id == *child_id);
            let waiting = invocation.waiters.iter().find_map(|(resource, waiters)| {
                waiters
                    .iter()
                    .any(|request| request.child_id == *child_id)
                    .then_some(*resource)
            });
            ChildDiagnostic {
                child_id: *child_id,
                parent_child_id: *parent,
                order: *order,
                phase: if active.is_some() {
                    "operating"
                } else if waiting.is_some() {
                    "waiting"
                } else {
                    "lua"
                }
                .to_string(),
                current_function: bridge.current_functions.get(child_id).cloned(),
                current_operation: active.map(|active| active.request.operation.name().to_string()),
                held_resource: active.and_then(|active| active.request.operation.resource()),
                waiting_resource: waiting,
            }
        })
        .collect();
    if let Some(script) = invocation.status.script.as_mut() {
        script.current_operation = invocation.diagnostics.current_operation.clone();
        script.current_function = invocation.diagnostics.current_lua_function.clone();
        script.held_resources = invocation
            .diagnostics
            .held_resources
            .keys()
            .map(|resource| format!("{resource:?}").to_lowercase())
            .collect();
        script.waiting_resources = invocation
            .diagnostics
            .waiting_resources
            .keys()
            .map(|resource| format!("{resource:?}").to_lowercase())
            .collect();
        script.active_children = invocation.diagnostics.active_together_children.len() as u32;
    }
}

fn finish_status(invocation: &mut Invocation, now_ms: u64) {
    if invocation.status.phase == SkillPhase::Terminal {
        return;
    }
    let (outcome, reason, detail, result) = match invocation.result.as_ref() {
        Some(Ok(value)) => (SkillOutcome::Completed, None, None, Some(value.clone())),
        Some(Err(failure)) => (
            failure.outcome,
            Some(failure.message.clone()),
            Some(failure.clone()),
            None,
        ),
        None => return,
    };
    invocation.status.phase = SkillPhase::Terminal;
    invocation.status.outcome = Some(outcome);
    invocation.status.reason = reason;
    invocation.status.updated_at_ms = now_ms;
    invocation.status.progress = (outcome == SkillOutcome::Completed)
        .then_some(1.0)
        .or_else(|| skill_progress(invocation));
    invocation.diagnostics.phase = "terminal".to_string();
    invocation.diagnostics.terminal_outcome = Some(outcome);
    invocation.diagnostics.terminal_detail = detail.clone();
    let completed = SkillTraceEvent::Completed {
        at_ms: now_ms,
        outcome,
        detail,
        duration_ms: now_ms.saturating_sub(invocation.started_at_ms),
        result,
    };
    push_invocation_trace(invocation, completed);
}

fn external_preemption(events: &EventBatch) -> Option<SkillFailure> {
    for event in &events.events {
        let event_json = serde_json::to_value(event).unwrap_or(Value::Null);
        match event.kind {
            CockpitEventKind::ContactWithdrawalStarted => {
                return Some(SkillFailure::safety(
                    "contact_withdrawal",
                    "Brainstem contact-withdrawal reflex preempted the operation",
                    event_json,
                ));
            }
            CockpitEventKind::SafetyTripped | CockpitEventKind::EStopLatched => {
                return Some(SkillFailure::safety(
                    "brainstem_safety",
                    "Brainstem safety preempted the operation",
                    event_json,
                ));
            }
            CockpitEventKind::SessionReplaced | CockpitEventKind::AuthorityChanged => {
                return Some(SkillFailure::new(
                    SkillOutcome::AuthorityLost,
                    "authority_lost",
                    "possession authority changed while the skill was active",
                ));
            }
            CockpitEventKind::SessionClosed
            | CockpitEventKind::PeerRebootDetected
            | CockpitEventKind::HeartbeatExpired => {
                return Some(SkillFailure::new(
                    SkillOutcome::TransportLost,
                    "transport_lost",
                    "body transport or supervised heartbeat was lost",
                ));
            }
            _ => {}
        }
    }
    None
}
