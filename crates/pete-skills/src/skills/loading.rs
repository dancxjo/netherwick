fn load_skill_set(config: &LuaSkillConfig) -> Result<LoadedSkillSet> {
    let mut paths = discover_lua_files(&config.directory)?;
    paths.sort();
    anyhow::ensure!(
        !paths.is_empty(),
        "no Lua skills found in {}",
        config.directory.display()
    );
    let loaded_at_ms = wall_time_ms();
    let runtime_version = "Lua 5.4 / mlua 0.11.6".to_string();
    let mut skills = BTreeMap::new();
    let mut ordered_sources = Vec::new();
    let mut generation_hasher = Sha256::new();
    for path in paths {
        let source = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read Lua skill {}", path.display()))?;
        let function_name = path
            .file_stem()
            .and_then(|value| value.to_str())
            .context("Lua skill filename is not valid UTF-8")?
            .to_string();
        validate_identifier(&function_name)?;
        let source_hash = hex_sha256(source.as_bytes());
        generation_hasher.update(path.to_string_lossy().as_bytes());
        generation_hasher.update(source_hash.as_bytes());
        let skill_id = format!("{}.{}", config.namespace, function_name);
        anyhow::ensure!(
            !skills.contains_key(&skill_id),
            "duplicate Lua skill ID {skill_id}"
        );
        let loaded = SkillSource {
            metadata: LoadedSkill {
                skill_id: skill_id.clone(),
                function_name,
                source_path: path,
                source_hash,
                loaded_at_ms,
                runtime_version: runtime_version.clone(),
            },
            source: Arc::from(source),
        };
        skills.insert(skill_id, loaded.clone());
        ordered_sources.push(loaded);
    }
    let set = LoadedSkillSet {
        generation_hash: format!("{:x}", generation_hasher.finalize()),
        skills,
        ordered_sources,
    };
    validate_skill_set(config, &set)?;
    Ok(set)
}

fn discover_lua_files(directory: &Path) -> Result<Vec<PathBuf>> {
    fn visit(path: &Path, found: &mut Vec<PathBuf>) -> Result<()> {
        for entry in std::fs::read_dir(path)
            .with_context(|| format!("failed to read skill directory {}", path.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            let kind = entry.file_type()?;
            if kind.is_dir() {
                visit(&path, found)?;
            } else if kind.is_file() && path.extension().is_some_and(|ext| ext == "lua") {
                found.push(path);
            }
        }
        Ok(())
    }
    let mut found = Vec::new();
    visit(directory, &mut found)?;
    Ok(found)
}

fn validate_skill_set(config: &LuaSkillConfig, set: &LoadedSkillSet) -> Result<()> {
    let bridge = Arc::new(Bridge::default());
    bridge.state.lock().expect("skill bridge lock").snapshot =
        Some(Now::blank(0, pete_body::BodySense::default()));
    let lua = build_vm(config, set, bridge)?;
    for source in set.skills.values() {
        let value: LuaValue = lua
            .globals()
            .get(source.metadata.function_name.as_str())
            .with_context(|| {
                format!(
                    "{} did not export {}",
                    source.metadata.source_path.display(),
                    source.metadata.function_name
                )
            })?;
        anyhow::ensure!(
            matches!(value, LuaValue::Function(_)),
            "{} must export function {}",
            source.metadata.source_path.display(),
            source.metadata.function_name
        );
    }
    Ok(())
}

fn build_vm(config: &LuaSkillConfig, set: &LoadedSkillSet, bridge: Arc<Bridge>) -> Result<Lua> {
    let libraries = StdLib::TABLE | StdLib::STRING | StdLib::MATH | StdLib::UTF8;
    let lua = Lua::new_with(libraries, LuaOptions::default())?;
    lua.set_memory_limit(config.memory_limit_bytes)?;
    let remaining = Arc::new(AtomicU64::new(config.instruction_budget));
    let hook_remaining = remaining.clone();
    let hook_bridge = bridge.clone();
    // This must be global: ordinary skills, `try`, `carefully`, and every
    // `together` child execute in newly-created Lua threads.
    lua.set_global_hook(
        HookTriggers::new()
            .every_nth_instruction(1_000)
            .on_calls()
            .on_returns(),
        move |_, debug| {
            match debug.event() {
                DebugEvent::Count => {
                    let prior = hook_remaining.fetch_sub(1_000, Ordering::Relaxed);
                    if prior <= 1_000 {
                        return Err(SkillFailure::new(
                            SkillOutcome::BudgetExceeded,
                            "instruction_budget_exceeded",
                            "Lua instruction budget exhausted",
                        )
                        .encoded());
                    }
                }
                DebugEvent::Call | DebugEvent::TailCall => {
                    if let Some(name) = debug.names().name {
                        let mut state = hook_bridge.state.lock().expect("skill bridge lock");
                        let child = state.current_child;
                        state.current_functions.insert(child, name.into_owned());
                    }
                }
                DebugEvent::Ret => {
                    let mut state = hook_bridge.state.lock().expect("skill bridge lock");
                    let child = state.current_child;
                    state.current_functions.remove(&child);
                }
                _ => {}
            }
            Ok(VmState::Continue)
        },
    )?;
    install_api(&lua, bridge, config.maximum_operation_ms)?;
    for source in &set.ordered_sources {
        lua.load(source.source.as_ref())
            .set_name(source.metadata.source_path.to_string_lossy())
            .exec()
            .with_context(|| {
                format!(
                    "failed to load Lua skill {}",
                    source.metadata.source_path.display()
                )
            })?;
    }
    remove_forbidden_globals(&lua)?;
    Ok(lua)
}

fn remove_forbidden_globals(lua: &Lua) -> mlua::Result<()> {
    let globals = lua.globals();
    for name in [
        "io",
        "os",
        "debug",
        "package",
        "dofile",
        "loadfile",
        "load",
        "collectgarbage",
        "coroutine",
        "rawget",
        "rawset",
        "rawequal",
        "setmetatable",
        "getmetatable",
    ] {
        globals.set(name, LuaValue::Nil)?;
    }
    let math: Table = globals.get("math")?;
    math.set("random", LuaValue::Nil)?;
    math.set("randomseed", LuaValue::Nil)?;
    Ok(())
}
