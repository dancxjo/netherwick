fn decode_lua_error(error: LuaError) -> SkillFailure {
    let rendered = error.to_string();
    if let Some(index) = rendered.find(LUA_ERROR_PREFIX) {
        let json = &rendered[index + LUA_ERROR_PREFIX.len()..];
        let json = json.lines().next().unwrap_or(json);
        if let Ok(failure) = serde_json::from_str(json) {
            return failure;
        }
    }
    if rendered.contains("memory error") || rendered.contains("not enough memory") {
        return SkillFailure::new(
            SkillOutcome::BudgetExceeded,
            "memory_budget_exceeded",
            "Lua memory budget exhausted",
        );
    }
    if rendered.contains("stack overflow") {
        return SkillFailure::new(
            SkillOutcome::BudgetExceeded,
            "stack_budget_exceeded",
            "Lua recursion or stack budget exhausted",
        );
    }
    SkillFailure::new(SkillOutcome::ScriptError, "script_error", rendered)
}

fn failure_to_lua(lua: &Lua, failure: &SkillFailure) -> mlua::Result<LuaValue> {
    lua.to_value(failure)
}

fn parse_hazard(value: &LuaValue) -> mlua::Result<HazardKind> {
    match value {
        LuaValue::String(value) => match value.to_str()?.as_ref() {
            "bumper_front" | "bump" | "contact" => Ok(HazardKind::BumperFront),
            "cliff" => Ok(HazardKind::Cliff),
            other => Err(LuaError::RuntimeError(format!(
                "unsupported CAREFUL hazard {other}"
            ))),
        },
        LuaValue::Table(table) => {
            let kind: String = table.get("kind")?;
            match kind.as_str() {
                "bumper_front" | "bump" | "contact" => Ok(HazardKind::BumperFront),
                "cliff" => Ok(HazardKind::Cliff),
                other => Err(LuaError::RuntimeError(format!(
                    "unsupported CAREFUL hazard {other}"
                ))),
            }
        }
        _ => Err(LuaError::RuntimeError(
            "carefully requires a validated hazard name or handle".into(),
        )),
    }
}

fn validate_hazard_acknowledged(
    now: &Now,
    hazard: HazardKind,
) -> std::result::Result<(), SkillFailure> {
    let active = match hazard {
        HazardKind::BumperFront => now.body.flags.bump_left || now.body.flags.bump_right,
        HazardKind::Cliff => QueryPredicate::Cliff.evaluate(now),
    };
    if active {
        Ok(())
    } else {
        Err(SkillFailure::new(
            SkillOutcome::Failed,
            "hazard_not_acknowledged",
            format!("{hazard:?} is not active in the current Now"),
        ))
    }
}

fn request_arguments(request: &SkillRequest, now: &Now) -> Value {
    let target = request.target.as_ref().map(|target| {
        let id = target.0.clone();
        let observation = now
            .objects
            .observations
            .iter()
            .find(|observation| stable_observation_id(observation) == id);
        json!({
            "id": id,
            "kind": observation.map(|value| object_class_name(&value.class)).unwrap_or("unknown"),
            "label": observation.map(|value| value.label.as_str()).unwrap_or("target"),
        })
    });
    json!({
        "goal_id": request.goal_id.as_ref().map(|value| value.as_str()),
        "implementation_id": request.implementation_id,
        "behavior_id": request.behavior_id,
        "target": target,
        "bearing_rad": request.bearing_rad,
        "angle_rad": request.bearing_rad,
        "range_m": request.range_m,
        "distance_m": request.range_m,
        "stop_range_m": request.stop_range_m,
        "maximum_duration_ms": request.maximum_duration_ms,
        "expected_progress": request.expected_progress,
        "progress_metric": request.progress_metric,
    })
}

fn request_arguments_lua(lua: &Lua, request: &SkillRequest, now: &Now) -> mlua::Result<Table> {
    let table = lua.create_table()?;
    table.set(
        "goal_id",
        request.goal_id.as_ref().map(|value| value.as_str()),
    )?;
    table.set("implementation_id", request.implementation_id.clone())?;
    table.set("behavior_id", request.behavior_id.clone())?;
    table.set("bearing_rad", request.bearing_rad)?;
    table.set("angle_rad", request.bearing_rad)?;
    table.set("range_m", request.range_m)?;
    table.set("distance_m", request.range_m)?;
    table.set("stop_range_m", request.stop_range_m)?;
    table.set("maximum_duration_ms", request.maximum_duration_ms)?;
    table.set("expected_progress", request.expected_progress)?;
    table.set("progress_metric", request.progress_metric.clone())?;
    if let Some(target) = request.target.as_ref() {
        let observation =
            now.objects.observations.iter().find(|observation| {
                stable_observation_id(observation).eq_ignore_ascii_case(&target.0)
            });
        let world_entity = now
            .world
            .entities
            .iter()
            .find(|(id, _)| id.0.eq_ignore_ascii_case(&target.0))
            .map(|(_, entity)| entity);
        let social_person = now
            .world
            .social
            .people
            .iter()
            .find(|(person_id, _)| person_id.0.eq_ignore_ascii_case(&target.0))
            .map(|(_, person)| person);
        let kind = observation
            .map(|value| object_class_name(&value.class))
            .or_else(|| world_entity.map(|entity| world_entity_kind_name(&entity.kind)))
            .or_else(|| social_person.map(|_| "person"))
            .unwrap_or("unknown");
        let label = observation
            .map(|value| value.label.as_str())
            .or_else(|| world_entity.map(|entity| entity.label.as_str()))
            .or_else(|| {
                social_person
                    .and_then(|person| person.preferred_name.as_ref())
                    .map(|name| name.value.as_str())
            })
            .unwrap_or("target");
        table.set(
            "target",
            lua.create_userdata(
                EntityHandle::new(target.0.clone(), kind, label)
                    .with_name(recognized_person_name(now, &target.0)),
            )?,
        )?;
    }
    Ok(table)
}

fn function_name_for_skill(skill: SkillId) -> &'static str {
    match skill {
        SkillId::StopAndStabilize => "stopAndStabilize",
        SkillId::TurnTowardTarget => "turnTowardTarget",
        SkillId::FollowBearing => "followBearingSkill",
        SkillId::ApproachTarget => "approachTarget",
        SkillId::BackAway => "driveFor",
        SkillId::InspectTarget => "inspectObject",
        SkillId::WallFollow => "wallFollow",
        SkillId::AlignWithDock => "alignWithDockSkill",
        SkillId::SystematicSearch => "systematicSearch",
        SkillId::HoldHeading => "holdHeadingSkill",
        SkillId::RetreatFromCliff => "retreatFromCliff",
        SkillId::ReleasePersistentBumper => "releasePersistentBumper",
        SkillId::TurnBy => "turnBySkill",
        SkillId::DriveDistance => "driveDistanceSkill",
        SkillId::Undock => "undockSkill",
        SkillId::SearchForDock => "searchForDock",
        SkillId::ReturnToDock => "returnToDock",
        SkillId::RuntimeLoaded => unreachable!("runtime-loaded skills provide implementation_id"),
    }
}

fn intention_key(request: &SkillRequest) -> String {
    format!(
        "{:?}:{:?}:{:?}:{:?}",
        request.skill_id, request.goal_id, request.behavior_id, request.target
    )
}

fn bounded_now_for_trace(now: &Now) -> Value {
    json!({
        "t_ms": now.t_ms,
        "body": {
            "charging": now.body.charging,
            "flags": now.body.flags,
            "odometry": now.body.odometry,
            "infrared_character": now.body.infrared_character,
        },
        "objects": now.objects.observations.iter().take(32).collect::<Vec<_>>(),
        "active_goal": now.self_sense.active_goal,
    })
}

fn bounded_lua_value(lua: &Lua, value: LuaValue) -> mlua::Result<Value> {
    let value: Value = lua.from_value(value)?;
    bounded_value(value, MAX_CONVERTED_VALUE_BYTES)
        .map_err(|error| LuaError::RuntimeError(error.to_string()))
}

fn lua_values_to_json(lua: &Lua, values: MultiValue) -> Result<Value> {
    if values.is_empty() {
        return Ok(Value::Null);
    }
    if values.len() == 1 {
        return Ok(lua.from_value(values.into_iter().next().unwrap_or(LuaValue::Nil))?);
    }
    let mut converted = Vec::new();
    for value in values {
        converted.push(lua.from_value(value)?);
    }
    Ok(Value::Array(converted))
}

fn bounded_value(value: Value, maximum_bytes: usize) -> Result<Value> {
    let encoded = serde_json::to_vec(&value)?;
    anyhow::ensure!(
        encoded.len() <= maximum_bytes,
        "Lua value exceeds {maximum_bytes} bytes"
    );
    Ok(value)
}

fn push_invocation_trace(invocation: &mut Invocation, event: SkillTraceEvent) {
    if invocation.trace.len() == MAX_TRACE_EVENTS {
        invocation.trace.remove(0);
    }
    invocation.bridge.push_trace(event.clone());
    invocation.trace.push(event);
}

fn skill_progress(invocation: &mut Invocation) -> Option<f32> {
    let state = invocation.bridge.state.lock().expect("skill bridge lock");
    let metric = invocation.request.progress_metric.as_str();
    if let Some(reported) = state
        .reported_progress
        .get(metric)
        .copied()
        .or_else(|| state.reported_progress.get("goal_progress").copied())
    {
        return Some(reported.clamp(0.0, 1.0));
    }
    let current = state
        .progress
        .get(metric)
        .copied()
        .or_else(|| invocation.diagnostics.progress.get(metric).copied())?;
    match metric {
        "bearing_error" | "target_distance" => {
            let baseline = *invocation.metric_baseline.get_or_insert(current);
            if baseline <= f32::EPSILON {
                Some(if current <= invocation.request.progress_tolerance {
                    1.0
                } else {
                    0.0
                })
            } else {
                Some(((baseline - current) / baseline).clamp(0.0, 1.0))
            }
        }
        "reverse_displacement"
        | "uncertainty_reduction"
        | "path_progress"
        | "frontier_coverage"
        | "motion_stability" => Some(current.clamp(0.0, 1.0)),
        _ => Some(current.clamp(0.0, 1.0)),
    }
}

fn stable_observation_id(observation: &pete_now::ObjectObservation) -> String {
    format!(
        "{}:{}",
        object_class_name(&observation.class),
        observation.label
    )
}

fn current_entity_exists(now: &Now, target: &EntityHandle) -> bool {
    now.objects
        .observations
        .iter()
        .any(|observation| stable_observation_id(observation).eq_ignore_ascii_case(target.id()))
        || now
            .world
            .entities
            .keys()
            .any(|id| id.0.eq_ignore_ascii_case(target.id()))
        || now.world.social.people.iter().any(|(person_id, person)| {
            person_id.0.eq_ignore_ascii_case(target.id()) && person.presence.present
        })
}

fn current_entity_bearing(now: &Now, target: &EntityHandle) -> Option<f32> {
    now.objects
        .observations
        .iter()
        .find(|observation| stable_observation_id(observation).eq_ignore_ascii_case(target.id()))
        .map(|observation| observation.bearing_rad)
        .or_else(|| {
            now.world
                .entities
                .iter()
                .find(|(id, _)| id.0.eq_ignore_ascii_case(target.id()))
                .and_then(|(_, entity)| entity.bearing_rad)
        })
        .or_else(|| {
            now.world
                .social
                .people
                .iter()
                .find(|(person_id, _)| person_id.0.eq_ignore_ascii_case(target.id()))
                .and_then(|(_, person)| person.location.as_ref())
                .and_then(|location| location.bearing_rad)
        })
}

fn current_entity_distance(now: &Now, target: &EntityHandle) -> Option<f32> {
    now.objects
        .observations
        .iter()
        .find(|observation| stable_observation_id(observation).eq_ignore_ascii_case(target.id()))
        .and_then(|observation| observation.distance_m)
        .or_else(|| {
            now.world
                .entities
                .iter()
                .find(|(id, _)| id.0.eq_ignore_ascii_case(target.id()))
                .and_then(|(_, entity)| entity.distance_m)
        })
        .or_else(|| {
            now.world
                .social
                .people
                .iter()
                .find(|(person_id, _)| person_id.0.eq_ignore_ascii_case(target.id()))
                .and_then(|(_, person)| person.location.as_ref())
                .and_then(|location| location.distance_m)
        })
}

fn recognized_person_name(now: &Now, target_id: &str) -> Option<String> {
    now.world
        .social
        .people
        .iter()
        .find(|(person_id, _)| person_id.0.eq_ignore_ascii_case(target_id))
        .and_then(|(_, person)| {
            (!person.identity_is_uncertain())
                .then(|| {
                    person
                        .preferred_name
                        .as_ref()
                        .map(|name| name.value.clone())
                })
                .flatten()
        })
}

fn object_matches(class: &ObjectClass, label: &str, wanted: &str) -> bool {
    label.eq_ignore_ascii_case(wanted) || object_class_name(class).eq_ignore_ascii_case(wanted)
}

fn object_class_name(class: &ObjectClass) -> &'static str {
    match class {
        ObjectClass::Obstacle => "obstacle",
        ObjectClass::Charger => "dock",
        ObjectClass::Person => "person",
        ObjectClass::SoundSource => "sound",
        ObjectClass::Landmark => "landmark",
        ObjectClass::Unknown => "unknown",
    }
}

fn world_entity_kind_name(kind: &pete_now::WorldEntityKind) -> &'static str {
    match kind {
        pete_now::WorldEntityKind::Obstacle => "obstacle",
        pete_now::WorldEntityKind::Charger => "dock",
        pete_now::WorldEntityKind::Person => "person",
        pete_now::WorldEntityKind::SoundSource => "sound",
        pete_now::WorldEntityKind::Landmark => "landmark",
        pete_now::WorldEntityKind::Door => "door",
        pete_now::WorldEntityKind::Region => "region",
        pete_now::WorldEntityKind::Unknown => "unknown",
    }
}

fn validate_identifier(value: &str) -> Result<()> {
    let mut chars = value.chars();
    anyhow::ensure!(
        chars
            .next()
            .is_some_and(|value| value == '_' || value.is_ascii_alphabetic())
            && chars.all(|value| value == '_' || value.is_ascii_alphanumeric()),
        "{value:?} is not a valid Lua function identifier"
    );
    Ok(())
}

fn finite(value: f32, name: &str) -> mlua::Result<f32> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(LuaError::RuntimeError(format!("{name} must be finite")))
    }
}

fn bearing_from_lua_value(bridge: &Bridge, value: LuaValue) -> mlua::Result<f32> {
    match value {
        LuaValue::Integer(value) => finite(value as f32, "bearing"),
        LuaValue::Number(value) => finite(value as f32, "bearing"),
        LuaValue::UserData(target) => {
            let target = target.borrow::<EntityHandle>()?;
            let now = bridge.snapshot()?;
            current_entity_bearing(&now, &target).ok_or_else(|| {
                if current_entity_exists(&now, &target) {
                    SkillFailure::new(
                        SkillOutcome::Failed,
                        "bearing_unavailable",
                        format!("entity {} has no current bearing", target.id),
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
        }
        _ => Err(LuaError::RuntimeError(
            "face/turnToward requires a bearing or entity handle".to_string(),
        )),
    }
}

fn make_operation<A, F>(args: A, make: F) -> mlua::Result<HostOperation>
where
    F: FnOnce(A) -> mlua::Result<HostOperation>,
{
    make(args)
}

fn anyhow_lua(condition: bool, message: &str) -> mlua::Result<()> {
    if condition {
        Ok(())
    } else {
        Err(LuaError::RuntimeError(message.to_string()))
    }
}

fn hex_sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn wall_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
