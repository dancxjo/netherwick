fn install_api(lua: &Lua, bridge: Arc<Bridge>, maximum_operation_ms: u64) -> mlua::Result<()> {
    install_operations(lua, bridge.clone(), maximum_operation_ms)?;
    install_queries(lua, bridge.clone())?;
    install_composition(lua, bridge.clone())?;
    install_provenance(lua, bridge)?;
    Ok(())
}

fn install_operations(
    lua: &Lua,
    bridge: Arc<Bridge>,
    maximum_operation_ms: u64,
) -> mlua::Result<()> {
    macro_rules! async_op {
        ($name:literal, $args:ty, $make:expr) => {{
            let bridge = bridge.clone();
            let make: Arc<dyn Fn($args) -> mlua::Result<HostOperation> + Send + Sync + 'static> =
                Arc::new($make);
            lua.globals().set(
                $name,
                lua.create_async_function(move |lua, args: $args| {
                    let bridge = bridge.clone();
                    let make = make.clone();
                    async move {
                        if bridge.cancelled.load(Ordering::Acquire) {
                            return Err(SkillFailure::new(
                                SkillOutcome::Cancelled,
                                "cancelled",
                                "foreground skill was cancelled",
                            )
                            .encoded());
                        }
                        let operation = make_operation(args, |args| make(args))?;
                        let value = bridge
                            .enqueue(operation, maximum_operation_ms)
                            .await
                            .map_err(|failure| failure.encoded())?;
                        lua.to_value(&value)
                    }
                })?,
            )?;
        }};
    }
    async_op!("stop", (), |_| Ok(HostOperation::Stop));
    async_op!("faceBearing", f32, |bearing| Ok(
        HostOperation::FaceBearing {
            bearing_rad: finite(bearing, "bearing")?,
        }
    ));
    async_op!("face", LuaValue, {
        let bridge = bridge.clone();
        move |target| {
            Ok(HostOperation::FaceBearing {
                bearing_rad: bearing_from_lua_value(&bridge, target)?,
            })
        }
    });
    async_op!("turn", f32, |angle| Ok(HostOperation::TurnBy {
        angle_rad: finite(angle, "angle")?,
    }));
    async_op!("turnBy", f32, |angle| Ok(HostOperation::TurnBy {
        angle_rad: finite(angle, "angle")?,
    }));
    async_op!("turnToward", LuaValue, {
        let bridge = bridge.clone();
        move |target| {
            Ok(HostOperation::FaceBearing {
                bearing_rad: bearing_from_lua_value(&bridge, target)?,
            })
        }
    });
    async_op!("drive", (f32, u64), |(velocity, duration_ms)| Ok(
        HostOperation::Drive {
            linear_m_s: finite(velocity, "velocity")?.clamp(-0.12, 0.12),
            duration_ms: duration_ms.min(30_000),
        }
    ));
    async_op!("driveDistance", (f32, Option<f32>), |(
        distance,
        velocity,
    )| {
        let distance = finite(distance, "distance")?.clamp(-5.0, 5.0);
        let direction = if distance < 0.0 { -1.0 } else { 1.0 };
        Ok(HostOperation::DriveDistance {
            distance_m: distance,
            velocity_m_s: finite(velocity.unwrap_or(0.08), "velocity")?
                .abs()
                .clamp(0.01, 0.12)
                * direction,
        })
    });
    async_op!("approach", (mlua::AnyUserData, Option<f32>), |(
        target,
        stop,
    )| {
        Ok(HostOperation::Approach {
            target: target.borrow::<EntityHandle>()?.clone(),
            stop_range_m: finite(stop.unwrap_or(0.30), "stop range")?.clamp(0.05, 5.0),
        })
    });
    async_op!("followBearing", (f32, Option<f32>), |(
        bearing,
        velocity,
    )| {
        Ok(HostOperation::FollowBearing {
            bearing_rad: finite(bearing, "bearing")?,
            linear_m_s: finite(velocity.unwrap_or(0.045), "velocity")?.clamp(0.0, 0.12),
        })
    });
    async_op!("holdHeading", (f32, Option<f32>), |(heading, velocity)| {
        Ok(HostOperation::HoldHeading {
            heading_rad: finite(heading, "heading")?,
            linear_m_s: finite(velocity.unwrap_or(0.0), "velocity")?.clamp(-0.12, 0.12),
        })
    });
    async_op!("followWall", (Option<String>, Option<f32>), |(
        side,
        distance,
    )| {
        let side = side.unwrap_or_else(|| "left".to_string());
        if side != "left" && side != "right" {
            return Err(LuaError::RuntimeError(
                "wall side must be left or right".into(),
            ));
        }
        Ok(HostOperation::FollowWall {
            side,
            distance_m: finite(distance.unwrap_or(0.25), "wall distance")?.clamp(0.05, 2.0),
        })
    });
    async_op!("scan", (), |_| Ok(HostOperation::Scan));
    async_op!("lookAt", mlua::AnyUserData, |target| {
        Ok(HostOperation::LookAt {
            target: target.borrow::<EntityHandle>()?.clone(),
        })
    });
    async_op!("observe", mlua::AnyUserData, |target| {
        Ok(HostOperation::Observe {
            target: target.borrow::<EntityHandle>()?.clone(),
        })
    });
    async_op!("searchForDockSignal", (), |_| Ok(
        HostOperation::SearchForDockSignal
    ));
    async_op!("alignWithDock", (), |_| Ok(HostOperation::AlignWithDock));
    async_op!("verifyCharging", (), |_| Ok(HostOperation::VerifyCharging));
    async_op!("undock", (), |_| Ok(HostOperation::Undock));
    async_op!("retreatFromContact", Option<f32>, |distance_mm| Ok(
        HostOperation::Retreat {
            hazard: HazardKind::BumperFront,
            distance_m: finite(distance_mm.unwrap_or(100.0), "distance")?.clamp(1.0, 500.0)
                / 1_000.0,
        }
    ));
    async_op!("retreatFromCliff", Option<f32>, |distance_mm| Ok(
        HostOperation::Retreat {
            hazard: HazardKind::Cliff,
            distance_m: finite(distance_mm.unwrap_or(100.0), "distance")?.clamp(1.0, 500.0)
                / 1_000.0,
        }
    ));
    async_op!("retreat", Option<f32>, {
        let bridge = bridge.clone();
        move |distance_mm| {
            let child = bridge.current_child();
            let hazard = bridge
                .state
                .lock()
                .expect("skill bridge lock")
                .child_hazard
                .get(&child)
                .copied()
                .ok_or_else(|| {
                    LuaError::RuntimeError(
                        "retreat must be called inside carefully(hazard, function)".into(),
                    )
                })?;
            Ok(HostOperation::Retreat {
                hazard,
                distance_m: finite(distance_mm.unwrap_or(100.0), "distance")?.clamp(1.0, 500.0)
                    / 1_000.0,
            })
        }
    });
    async_op!("completeHazardRecovery", String, |hazard| {
        let hazard = match hazard.as_str() {
            "bumper_front" | "bump" | "contact" => HazardKind::BumperFront,
            "cliff" => HazardKind::Cliff,
            _ => {
                return Err(LuaError::RuntimeError(
                    "completeHazardRecovery requires bumper_front or cliff".to_string(),
                ))
            }
        };
        Ok(HostOperation::CompleteHazardRecovery { hazard })
    });
    async_op!("releasePersistentBumper", (), |_| Ok(
        HostOperation::ReleasePersistentBumper
    ));
    async_op!("grasp", mlua::AnyUserData, |target| {
        Ok(HostOperation::Grasp {
            target: target.borrow::<EntityHandle>()?.clone(),
        })
    });
    async_op!("release", Option<mlua::AnyUserData>, |target| {
        Ok(HostOperation::Release {
            target: target
                .map(|target| target.borrow::<EntityHandle>().map(|value| value.clone()))
                .transpose()?,
        })
    });
    async_op!("bringToMouth", mlua::AnyUserData, |target| {
        Ok(HostOperation::BringToMouth {
            target: target.borrow::<EntityHandle>()?.clone(),
        })
    });
    async_op!("chew", (), |_| Ok(HostOperation::Chew));
    async_op!("swallow", (), |_| Ok(HostOperation::Swallow));
    async_op!("say", String, |text| {
        anyhow_lua(text.len() <= 4_096, "speech exceeds 4096 bytes")?;
        Ok(HostOperation::Say { text })
    });
    async_op!("playFeedback", String, |pattern| {
        anyhow_lua(pattern.len() <= 128, "feedback pattern is too long")?;
        Ok(HostOperation::PlayFeedback { pattern })
    });
    async_op!("waitUntil", (String, Option<u64>), |(
        predicate,
        timeout,
    )| {
        Ok(HostOperation::WaitUntil {
            predicate,
            timeout_ms: timeout.unwrap_or(5_000).clamp(1, 30_000),
        })
    });
    Ok(())
}

fn install_queries(lua: &Lua, bridge: Arc<Bridge>) -> mlua::Result<()> {
    let query_bridge = bridge.clone();
    lua.globals().set(
        "nearestVisible",
        lua.create_function(move |lua, kind: String| {
            let now = query_bridge.snapshot()?;
            let observation = now
                .objects
                .observations
                .iter()
                .filter(|observation| object_matches(&observation.class, &observation.label, &kind))
                .min_by(|left, right| {
                    left.distance_m
                        .unwrap_or(f32::INFINITY)
                        .total_cmp(&right.distance_m.unwrap_or(f32::INFINITY))
                });
            observation
                .map(|observation| {
                    lua.create_userdata(EntityHandle::new(
                        stable_observation_id(observation),
                        object_class_name(&observation.class),
                        observation.label.clone(),
                    ))
                })
                .transpose()
        })?,
    )?;
    let query_bridge = bridge.clone();
    lua.globals().set(
        "visible",
        lua.create_function(move |lua, kind: String| {
            let now = query_bridge.snapshot()?;
            let values = lua.create_table()?;
            for (index, observation) in now
                .objects
                .observations
                .iter()
                .filter(|observation| object_matches(&observation.class, &observation.label, &kind))
                .take(128)
                .enumerate()
            {
                values.set(
                    index + 1,
                    lua.create_userdata(
                        EntityHandle::new(
                            stable_observation_id(observation),
                            object_class_name(&observation.class),
                            observation.label.clone(),
                        )
                        .with_name(recognized_person_name(
                            &now,
                            &stable_observation_id(observation),
                        )),
                    )?,
                )?;
            }
            Ok(values)
        })?,
    )?;
    for (name, bearing) in [("distanceTo", false), ("bearingTo", true)] {
        let query_bridge = bridge.clone();
        lua.globals().set(
            name,
            lua.create_function(move |_, target: mlua::AnyUserData| {
                let target = target.borrow::<EntityHandle>()?;
                let now = query_bridge.snapshot()?;
                let value = if bearing {
                    current_entity_bearing(&now, &target)
                } else {
                    current_entity_distance(&now, &target)
                };
                value.ok_or_else(|| {
                    if current_entity_exists(&now, &target) {
                        SkillFailure::new(
                            SkillOutcome::Failed,
                            if bearing {
                                "bearing_unavailable"
                            } else {
                                "range_unavailable"
                            },
                            format!(
                                "entity {} has no current {}",
                                target.id,
                                if bearing { "bearing" } else { "range" }
                            ),
                        )
                        .encoded()
                    } else {
                        SkillFailure::new(
                            SkillOutcome::Failed,
                            "target_stale",
                            format!("entity {} is no longer visible", target.id),
                        )
                        .encoded()
                    }
                })
            })?,
        )?;
    }
    for (name, predicate) in [
        ("contactActive", QueryPredicate::Contact),
        ("cliffActive", QueryPredicate::Cliff),
        ("charging", QueryPredicate::Charging),
        ("cliffIsClear", QueryPredicate::CliffClear),
    ] {
        let query_bridge = bridge.clone();
        lua.globals().set(
            name,
            lua.create_function(move |_, ()| {
                let now = query_bridge.snapshot()?;
                Ok(predicate.evaluate(&now))
            })?,
        )?;
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum QueryPredicate {
    Contact,
    Cliff,
    Charging,
    CliffClear,
}

impl QueryPredicate {
    fn evaluate(self, now: &Now) -> bool {
        match self {
            Self::Contact => now.body.flags.bump_left || now.body.flags.bump_right,
            Self::Cliff => {
                now.body.flags.cliff_left
                    || now.body.flags.cliff_front_left
                    || now.body.flags.cliff_front_right
                    || now.body.flags.cliff_right
            }
            Self::Charging => now.body.charging,
            Self::CliffClear => !Self::Cliff.evaluate(now),
        }
    }
}

fn install_composition(lua: &Lua, bridge: Arc<Bridge>) -> mlua::Result<()> {
    let together_bridge = bridge.clone();
    lua.globals().set(
        "together",
        lua.create_async_function(move |lua, functions: Variadic<Function>| {
            let bridge = together_bridge.clone();
            async move {
                anyhow_lua(
                    !functions.is_empty(),
                    "together requires at least one function",
                )?;
                anyhow_lua(
                    functions.len() <= 32,
                    "together supports at most 32 children",
                )?;
                let (parent, ids) = bridge.allocate_children(functions.len());
                let now_ms = bridge.snapshot()?.t_ms;
                let mut threads = Vec::with_capacity(functions.len());
                for (order, (function, child_id)) in
                    functions.into_iter().zip(ids.iter().copied()).enumerate()
                {
                    bridge.push_trace(SkillTraceEvent::ChildStarted {
                        at_ms: now_ms,
                        child_id,
                        parent_child_id: parent,
                        order,
                    });
                    threads.push(Some(Box::pin(
                        lua.create_thread(function)?.into_async::<MultiValue>(())?,
                    )));
                }
                let mut results: Vec<Option<MultiValue>> = vec![None; threads.len()];
                let outcome = std::future::poll_fn(|cx| {
                    let mut all_done = true;
                    for index in 0..threads.len() {
                        let Some(thread) = threads[index].as_mut() else {
                            continue;
                        };
                        all_done = false;
                        let previous = bridge.set_current_child(ids[index]);
                        let polled = thread.as_mut().poll(cx);
                        bridge.set_current_child(previous);
                        match polled {
                            Poll::Ready(Ok(values)) => {
                                results[index] = Some(values);
                                threads[index] = None;
                            }
                            Poll::Ready(Err(error)) => {
                                let failure = decode_lua_error(error);
                                bridge.cancel_children(
                                    ids.iter().copied().filter(|id| *id != ids[index]),
                                    failure.clone(),
                                );
                                return Poll::Ready(Err(failure.encoded()));
                            }
                            Poll::Pending => {}
                        }
                    }
                    if all_done || threads.iter().all(Option::is_none) {
                        Poll::Ready(Ok(()))
                    } else {
                        Poll::Pending
                    }
                })
                .await;
                bridge.set_current_child(parent);
                bridge.finish_children(&ids);
                outcome?;
                let table = lua.create_table()?;
                for (index, values) in results.into_iter().enumerate() {
                    let child_results = lua.create_table()?;
                    for (value_index, value) in values.unwrap_or_default().into_iter().enumerate() {
                        child_results.set(value_index + 1, value)?;
                    }
                    table.set(index + 1, child_results)?;
                }
                Ok(table)
            }
        })?,
    )?;

    let try_bridge = bridge.clone();
    lua.globals().set(
        "try",
        lua.create_async_function(move |lua, function: Function| {
            let bridge = try_bridge.clone();
            async move {
                let child = bridge.current_child();
                let previous = bridge.set_current_child(child);
                let result = lua
                    .create_thread(function)?
                    .into_async::<MultiValue>(())?
                    .await;
                bridge.set_current_child(previous);
                match result {
                    Ok(values) => {
                        let value = if values.len() == 1 {
                            values.into_iter().next().unwrap_or(LuaValue::Nil)
                        } else {
                            let table = lua.create_table()?;
                            for (index, value) in values.into_iter().enumerate() {
                                table.set(index + 1, value)?;
                            }
                            LuaValue::Table(table)
                        };
                        Ok((true, value))
                    }
                    Err(error) => {
                        let failure = decode_lua_error(error);
                        Ok((false, failure_to_lua(&lua, &failure)?))
                    }
                }
            }
        })?,
    )?;

    lua.globals().set(
        "blocked",
        lua.create_function(|_, value: LuaValue| Ok(value))?,
    )?;
    lua.globals().set(
        "error",
        lua.create_function(|lua, value: LuaValue| -> mlua::Result<()> {
            if let Ok(failure) = lua.from_value::<SkillFailure>(value.clone()) {
                return Err(failure.encoded());
            }
            let message = match value {
                LuaValue::String(value) => value.to_string_lossy(),
                other => format!("{other:?}"),
            };
            Err(LuaError::RuntimeError(message))
        })?,
    )?;
    lua.globals().set(
        "require",
        lua.create_function(|_, (condition, message): (bool, Option<String>)| {
            if condition {
                Ok(true)
            } else {
                Err(SkillFailure::new(
                    SkillOutcome::PostconditionFailed,
                    "postcondition_failed",
                    message.unwrap_or_else(|| "required postcondition was false".to_string()),
                )
                .encoded())
            }
        })?,
    )?;
    let careful_bridge = bridge;
    lua.globals().set(
        "carefully",
        lua.create_async_function(move |lua, (hazard, function): (LuaValue, Function)| {
            let bridge = careful_bridge.clone();
            async move {
                let child = bridge.current_child();
                let hazard = parse_hazard(&hazard)?;
                validate_hazard_acknowledged(&bridge.snapshot()?, hazard)
                    .map_err(|failure| failure.encoded())?;
                bridge
                    .state
                    .lock()
                    .expect("skill bridge lock")
                    .child_hazard
                    .insert(child, hazard);
                let result = lua
                    .create_thread(function)?
                    .into_async::<MultiValue>(())?
                    .await;
                bridge
                    .state
                    .lock()
                    .expect("skill bridge lock")
                    .child_hazard
                    .remove(&child);
                result
            }
        })?,
    )?;
    Ok(())
}

fn install_provenance(lua: &Lua, bridge: Arc<Bridge>) -> mlua::Result<()> {
    let progress_bridge = bridge.clone();
    lua.globals().set(
        "reportProgress",
        lua.create_function(move |_, (key, value): (String, f32)| {
            anyhow_lua(key.len() <= 128, "progress key is too long")?;
            let value = finite(value, "progress")?;
            let mut state = progress_bridge.state.lock().expect("skill bridge lock");
            anyhow_lua(
                state.reported_progress.contains_key(&key)
                    || state.reported_progress.len() < MAX_PROGRESS_ENTRIES,
                "too many progress values",
            )?;
            state
                .reported_progress
                .insert(key.clone(), value.clamp(0.0, 1.0));
            let at_ms = state.snapshot.as_ref().map_or(0, |now| now.t_ms);
            state
                .trace
                .push_back(SkillTraceEvent::Progress { at_ms, key, value });
            Ok(())
        })?,
    )?;
    let observation_bridge = bridge.clone();
    lua.globals().set(
        "hypothesize",
        lua.create_function(move |lua, (kind, value): (String, LuaValue)| {
            let value = bounded_lua_value(&lua, value)?;
            observation_bridge
                .state
                .lock()
                .expect("skill bridge lock")
                .observations
                .push(json!({
                    "kind": kind,
                    "value": value,
                    "provenance": "lua_skill",
                }));
            Ok(())
        })?,
    )?;
    let remember_bridge = bridge.clone();
    lua.globals().set(
        "remember",
        lua.create_function(move |lua, (key, value): (String, LuaValue)| {
            anyhow_lua(key.len() <= 128, "memory key is too long")?;
            let value = bounded_lua_value(&lua, value)?;
            remember_bridge
                .state
                .lock()
                .expect("skill bridge lock")
                .memories
                .push(json!({
                    "key": key,
                    "value": value,
                    "provenance": "lua_skill",
                }));
            Ok(())
        })?,
    )?;
    lua.globals().set(
        "acknowledge",
        lua.create_function(move |_, target: mlua::AnyUserData| {
            let target = target.borrow::<EntityHandle>()?.clone();
            let mut state = bridge.state.lock().expect("skill bridge lock");
            let now = state.snapshot.clone().ok_or_else(|| {
                LuaError::RuntimeError("canonical Now is unavailable".to_string())
            })?;
            let interaction = now
                .world
                .social
                .active_interaction
                .as_ref()
                .ok_or_else(|| {
                    SkillFailure::new(
                        SkillOutcome::PostconditionFailed,
                        "encounter_ended",
                        "cannot acknowledge a person outside an active encounter",
                    )
                    .encoded()
                })?;
            let participant = interaction
                .participants
                .iter()
                .find(|participant| participant.0.eq_ignore_ascii_case(target.id()))
                .ok_or_else(|| {
                    SkillFailure::new(
                        SkillOutcome::PostconditionFailed,
                        "person_not_in_encounter",
                        format!(
                            "{} is not a participant in the active encounter",
                            target.id()
                        ),
                    )
                    .encoded()
                })?;
            let acknowledgment_id = format!(
                "greet:{}:{}:{}",
                interaction.interaction_id.0, participant.0, state.execution_id
            );
            if !state.observations.iter().any(|observation| {
                observation
                    .pointer("/value/acknowledgment_id")
                    .and_then(Value::as_str)
                    == Some(acknowledgment_id.as_str())
            }) {
                state.observations.push(json!({
                    "kind": "social_acknowledgment",
                    "contract": "host_validated_social_acknowledgment_v1",
                    "value": {
                        "acknowledgment_id": acknowledgment_id,
                        "interaction_id": interaction.interaction_id.0,
                        "person_id": participant.0,
                        "occurred_at_ms": now.t_ms,
                    },
                    "provenance": "lua_skill",
                }));
            }
            Ok(true)
        })?,
    )?;
    Ok(())
}
