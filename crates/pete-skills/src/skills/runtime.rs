pub struct LuaSkillRuntime {
    config: LuaSkillConfig,
    active_set: Arc<LoadedSkillSet>,
    invocation: Option<Invocation>,
    next_execution_id: u64,
    attempts: BTreeMap<String, u32>,
    last_reload_error: Option<String>,
}

impl LuaSkillRuntime {
    pub fn load(config: LuaSkillConfig) -> Result<Self> {
        let active_set = Arc::new(load_skill_set(&config)?);
        Ok(Self {
            config,
            active_set,
            invocation: None,
            next_execution_id: 1,
            attempts: BTreeMap::new(),
            last_reload_error: None,
        })
    }

    pub fn discoverable_skills(&self) -> Vec<LoadedSkill> {
        self.active_set
            .skills
            .values()
            .map(|source| source.metadata.clone())
            .collect()
    }

    pub fn generation_hash(&self) -> &str {
        &self.active_set.generation_hash
    }

    pub fn last_reload_error(&self) -> Option<&str> {
        self.last_reload_error.as_deref()
    }

    pub fn reload(&mut self) -> Result<bool> {
        match load_skill_set(&self.config) {
            Ok(candidate) => {
                if candidate.generation_hash == self.active_set.generation_hash {
                    self.last_reload_error = None;
                    return Ok(false);
                }
                self.active_set = Arc::new(candidate);
                self.last_reload_error = None;
                Ok(true)
            }
            Err(error) => {
                self.last_reload_error = Some(error.to_string());
                Err(error)
            }
        }
    }

    pub fn is_active(&self) -> bool {
        self.invocation.is_some()
    }

    pub fn active_skill_id(&self) -> Option<&str> {
        self.invocation
            .as_ref()
            .map(|invocation| invocation.metadata.skill_id.as_str())
    }

    pub fn diagnostics(&self) -> SkillDiagnostic {
        self.invocation
            .as_ref()
            .map(|invocation| invocation.diagnostics.clone())
            .unwrap_or_else(|| SkillDiagnostic {
                phase: "idle".to_string(),
                ..SkillDiagnostic::default()
            })
    }

    pub fn trace(&self) -> Vec<SkillTraceEvent> {
        self.invocation
            .as_ref()
            .map(|invocation| invocation.trace.clone())
            .unwrap_or_default()
    }

    pub fn execution_record(&self) -> Option<SkillExecutionRecord> {
        let invocation = self.invocation.as_ref()?;
        let state = invocation.bridge.state.lock().expect("skill bridge lock");
        Some(SkillExecutionRecord {
            execution_id: invocation.status.execution_id,
            skill: invocation.metadata.clone(),
            request: invocation.request.clone(),
            diagnostics: invocation.diagnostics.clone(),
            trace: state.trace.iter().cloned().collect(),
            observations: state.observations.iter().take(128).cloned().collect(),
            memories: state.memories.iter().take(128).cloned().collect(),
        })
    }

    pub fn start(&mut self, request: SkillRequest, now: &Now) -> Result<()> {
        if self.invocation.is_some() {
            anyhow::bail!("a foreground Lua skill is already active");
        }
        let (skill_id, function_name) = if request.skill_id == SkillId::RuntimeLoaded {
            let skill_id = request
                .implementation_id
                .clone()
                .context("runtime-loaded skill request is missing implementation_id")?;
            let prefix = format!("{}.", self.config.namespace);
            let function_name = skill_id
                .strip_prefix(&prefix)
                .with_context(|| {
                    format!(
                        "runtime skill {skill_id} is outside configured namespace {}",
                        self.config.namespace
                    )
                })?
                .to_string();
            validate_identifier(&function_name)?;
            (skill_id, function_name)
        } else {
            let function_name = function_name_for_skill(request.skill_id).to_string();
            (
                format!("{}.{}", self.config.namespace, function_name),
                function_name,
            )
        };
        let source = self
            .active_set
            .skills
            .get(&skill_id)
            .with_context(|| format!("Lua skill {skill_id} is not loaded"))?
            .clone();
        let bridge = Arc::new(Bridge::default());
        bridge.state.lock().expect("skill bridge lock").snapshot = Some(now.clone());
        let lua = build_vm(&self.config, &self.active_set, bridge.clone())?;
        let function: Function = lua
            .globals()
            .get(function_name.as_str())
            .with_context(|| format!("loaded skill {skill_id} did not export {function_name}"))?;
        let arguments = request_arguments(&request, now);
        let lua_arguments = request_arguments_lua(&lua, &request, now)?;
        let thread = lua
            .create_thread(function)?
            .into_async::<MultiValue>(lua_arguments)?;
        let execution_id = self.next_execution_id;
        self.next_execution_id = execution_id.saturating_add(1);
        bridge.state.lock().expect("skill bridge lock").execution_id = execution_id;
        let metric_baseline = request.progress_baseline;
        let intention_key = intention_key(&request);
        let attempts = self.attempts.entry(intention_key).or_insert(0);
        *attempts = attempts.saturating_add(1);
        let status = SkillStatus {
            request: request.clone(),
            execution_id,
            phase: SkillPhase::Requested,
            outcome: None,
            progress: None,
            attempts: *attempts,
            dispatch_count: 0,
            started_at_ms: Some(now.t_ms),
            updated_at_ms: now.t_ms,
            reason: None,
            script: Some(pete_conductor::SkillScriptStatus {
                skill_id: skill_id.clone(),
                source_hash: source.metadata.source_hash.clone(),
                source_path: source.metadata.source_path.display().to_string(),
                current_function: Some(function_name.clone()),
                current_operation: None,
                held_resources: Vec::new(),
                waiting_resources: Vec::new(),
                active_children: 0,
            }),
        };
        let start = SkillTraceEvent::Started {
            at_ms: now.t_ms,
            skill_id: skill_id.clone(),
            source_hash: source.metadata.source_hash.clone(),
            arguments,
            starting_now: bounded_now_for_trace(now),
        };
        bridge.push_trace(start.clone());
        self.invocation = Some(Invocation {
            _lua: lua,
            _set: self.active_set.clone(),
            metadata: source.metadata.clone(),
            request,
            thread: Box::pin(thread),
            bridge,
            started_at_ms: now.t_ms,
            metric_baseline,
            active: HashMap::new(),
            owners: BTreeMap::new(),
            waiters: BTreeMap::new(),
            result: None,
            status,
            diagnostics: SkillDiagnostic {
                foreground_skill_id: Some(skill_id),
                source_hash: Some(source.metadata.source_hash),
                source_path: Some(source.metadata.source_path),
                start_time_ms: Some(now.t_ms),
                current_lua_function: Some(function_name),
                phase: "requested".to_string(),
                ..SkillDiagnostic::default()
            },
            trace: vec![start],
        });
        Ok(())
    }

    pub fn step<D: OrganDriver>(
        &mut self,
        now: &Now,
        events: &EventBatch,
        driver: &mut D,
    ) -> Option<SkillStatus> {
        let invocation = self.invocation.as_mut()?;
        invocation.status.updated_at_ms = now.t_ms;
        invocation.status.phase = SkillPhase::Running;
        invocation.diagnostics.phase = "running".to_string();
        invocation
            .bridge
            .state
            .lock()
            .expect("skill bridge lock")
            .snapshot = Some(now.clone());

        if invocation.request.maximum_duration_ms > 0
            && now.t_ms.saturating_sub(invocation.started_at_ms)
                >= invocation.request.maximum_duration_ms
        {
            cancel_invocation(
                invocation,
                driver,
                SkillFailure::new(
                    SkillOutcome::TimedOut,
                    "skill_timed_out",
                    "foreground skill exceeded its bounded duration",
                ),
            );
        }
        if let Some(failure) = external_preemption(events) {
            cancel_invocation(invocation, driver, failure);
        }

        service_child_cancellations(invocation, driver, now.t_ms);
        expire_resource_waits(invocation, now.t_ms);
        service_active_operations(
            invocation,
            driver,
            now,
            events,
            self.config.maximum_operation_ms,
        );
        grant_waiting_operations(invocation, now.t_ms);

        if invocation.result.is_none() {
            invocation.diagnostics.last_resume_ms = Some(now.t_ms);
            let previous = invocation.bridge.set_current_child(0);
            let poll_started = Instant::now();
            let waker = Waker::from(Arc::new(RuntimeWaker));
            let mut context = TaskContext::from_waker(&waker);
            let polled = catch_unwind(AssertUnwindSafe(|| {
                invocation.thread.as_mut().poll(&mut context)
            }));
            invocation.bridge.set_current_child(previous);
            if polled.is_err() {
                cancel_invocation(
                    invocation,
                    driver,
                    SkillFailure::new(
                        SkillOutcome::ScriptError,
                        "lua_vm_panic",
                        "embedded Lua VM panicked during activation",
                    ),
                );
            } else if poll_started.elapsed() > self.config.activation_budget {
                let failure = SkillFailure::new(
                    SkillOutcome::BudgetExceeded,
                    "wall_clock_budget_exceeded",
                    format!(
                        "Lua activation exceeded {} ms",
                        self.config.activation_budget.as_millis()
                    ),
                );
                cancel_invocation(invocation, driver, failure);
            } else {
                match polled.expect("Lua poll panic handled") {
                    Poll::Ready(Ok(values)) => {
                        let result = lua_values_to_json(&invocation._lua, values)
                            .and_then(|value| {
                                bounded_value(value, self.config.maximum_result_bytes)
                            })
                            .map_err(|error| {
                                SkillFailure::new(
                                    SkillOutcome::ScriptError,
                                    "result_conversion",
                                    error.to_string(),
                                )
                            });
                        invocation.result = Some(result);
                    }
                    Poll::Ready(Err(error)) => {
                        let failure = decode_lua_error(error);
                        cancel_invocation(invocation, driver, failure);
                    }
                    Poll::Pending => {
                        invocation.diagnostics.last_yield_ms = Some(now.t_ms);
                    }
                }
            }
        }

        drain_new_requests(invocation, now.t_ms);
        grant_waiting_operations(invocation, now.t_ms);
        update_diagnostics(invocation);

        if invocation.result.is_some() {
            finish_status(invocation, now.t_ms);
        } else {
            invocation.status.progress = skill_progress(invocation);
        }
        Some(invocation.status.clone())
    }

    pub fn cancel<D: OrganDriver>(
        &mut self,
        driver: &mut D,
        outcome: SkillOutcome,
        kind: impl Into<String>,
        message: impl Into<String>,
        now_ms: u64,
    ) -> Option<SkillStatus> {
        let invocation = self.invocation.as_mut()?;
        cancel_invocation(
            invocation,
            driver,
            SkillFailure::new(outcome, kind, message),
        );
        finish_status(invocation, now_ms);
        Some(invocation.status.clone())
    }

    pub fn take_terminal(&mut self) -> Option<(SkillStatus, Vec<SkillTraceEvent>)> {
        if !self
            .invocation
            .as_ref()
            .is_some_and(|invocation| invocation.status.phase == SkillPhase::Terminal)
        {
            return None;
        }
        let invocation = self.invocation.take()?;
        Some((invocation.status, invocation.trace))
    }

    pub fn shutdown<D: OrganDriver>(&mut self, driver: &mut D, now_ms: u64) {
        let _ = self.cancel(
            driver,
            SkillOutcome::Cancelled,
            "motherbrain_shutdown",
            "motherbrain shut down while the skill was active",
            now_ms,
        );
        driver.shutdown();
    }
}

struct RuntimeWaker;

impl Wake for RuntimeWaker {
    fn wake(self: Arc<Self>) {}
}
