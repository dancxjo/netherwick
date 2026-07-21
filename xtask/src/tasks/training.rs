fn train(args: &[String]) -> Result<()> {
    if args.first().is_some_and(|value| value == "--neat") {
        if args.len() != 2 {
            return fail("usage: just train --neat locomotion");
        }
        return train_neat(&args[1]);
    }
    if args.len() > 1 || args.first().is_some_and(|value| value != "virtual") {
        return fail("usage: just train virtual | just train --neat locomotion");
    }
    pete_tools(
        [
            "train",
            "virtual",
            "--ledger",
            &env_or("PETE_LEDGER", "data/ledger/virtual-live"),
            "--out-dir",
            &env_or("PETE_MODEL_OUT", "data/models/virtual/latest"),
            "--report-out",
            &env_or("PETE_REPORT_OUT", "data/reports/virtual/latest.json"),
            "--epochs",
            &env_or("PETE_EPOCHS", "5"),
        ],
        &[],
    )
}

fn train_neat(behavior: &str) -> Result<()> {
    let report_dir = env_or("PETE_NEAT_REPORT_DIR", "data/reports/neat/locomotion-v2");
    let migrated = PathBuf::from(&report_dir).join("trainer-state-schema3-leave-start-region.json");
    let state = env::var("PETE_NEAT_STATE_CHECKPOINT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            if migrated.is_file() {
                migrated
            } else {
                PathBuf::from(&report_dir).join("trainer-state.json")
            }
        });
    let resume = env_or("PETE_NEAT_RESUME", "");
    let founders = env_or("PETE_NEAT_FOUNDERS_REPORT", "");
    let start_stage = env_or("PETE_NEAT_START_STAGE", "");
    if !resume.is_empty() && !founders.is_empty() {
        return fail("PETE_NEAT_RESUME and PETE_NEAT_FOUNDERS_REPORT are mutually exclusive");
    }
    let mut extra = Vec::new();
    let continuation;
    if !resume.is_empty() {
        extra.extend(["--resume".to_owned(), resume.clone()]);
        continuation = PathBuf::from(resume);
        println!("NEAT continuation: resuming {}", continuation.display());
    } else if !founders.is_empty() {
        println!("NEAT continuation: reconstructing founders from {founders}");
        extra.extend(["--founders-report".to_owned(), founders]);
        extra.extend([
            "--start-stage".to_owned(),
            if start_stage.is_empty() {
                "leave-start-region".to_owned()
            } else {
                start_stage.clone()
            },
        ]);
        continuation = PathBuf::new();
    } else if !env_flag("PETE_NEAT_FRESH") && state.is_file() {
        extra.extend(["--resume".to_owned(), path(&state)?.to_owned()]);
        continuation = state.clone();
        println!("NEAT continuation: resuming {}", state.display());
    } else if !env_flag("PETE_NEAT_FRESH")
        && Path::new("data/reports/neat/locomotion/training-report.json").is_file()
    {
        let founders = "data/reports/neat/locomotion/training-report.json";
        println!("NEAT continuation: reconstructing founders from {founders}");
        extra.extend(["--founders-report".to_owned(), founders.to_owned()]);
        extra.extend([
            "--start-stage".to_owned(),
            if start_stage.is_empty() {
                "leave-start-region".to_owned()
            } else {
                start_stage.clone()
            },
        ]);
        continuation = PathBuf::new();
    } else {
        if !start_stage.is_empty() {
            extra.extend(["--start-stage".to_owned(), start_stage]);
        }
        continuation = PathBuf::new();
        println!("NEAT continuation: starting a fresh competence-gated run");
    }
    let generations = neat_generation_limit(
        &continuation,
        env::var("PETE_NEAT_GENERATIONS_PER_STAGE")
            .ok()
            .and_then(|value| value.parse().ok()),
        120,
        120,
    )
    .to_string();
    let mut values = vec![
        "neat-train".to_owned(),
        behavior.to_owned(),
        "--generations-per-stage".to_owned(),
        generations,
        "--population".to_owned(),
        env_or("PETE_NEAT_POPULATION", "64"),
        "--episodes-per-genome".to_owned(),
        env_or("PETE_NEAT_EPISODES_PER_GENOME", "12"),
        "--steps".to_owned(),
        env_or("PETE_NEAT_STEPS", "300"),
        "--transfer-episodes".to_owned(),
        env_or("PETE_NEAT_TRANSFER_EPISODES", "500"),
        "--seed".to_owned(),
        env_or("PETE_NEAT_SEED", "7"),
        "--heldout-seed".to_owned(),
        env_or("PETE_NEAT_HELDOUT_SEED", "9000001"),
        "--validation-seed".to_owned(),
        env_or("PETE_NEAT_VALIDATION_SEED", "8000001"),
        "--validation-every".to_owned(),
        env_or("PETE_NEAT_VALIDATION_EVERY", "4"),
        "--validation-passes".to_owned(),
        env_or("PETE_NEAT_VALIDATION_PASSES", "2"),
        "--compatibility-threshold".to_owned(),
        env_or("PETE_NEAT_COMPATIBILITY_THRESHOLD", "2.2"),
        "--compatibility-threshold-floor".to_owned(),
        env_or("PETE_NEAT_COMPATIBILITY_THRESHOLD_FLOOR", "0.05"),
        "--target-species-min".to_owned(),
        env_or("PETE_NEAT_TARGET_SPECIES_MIN", "4"),
        "--target-species-max".to_owned(),
        env_or("PETE_NEAT_TARGET_SPECIES_MAX", "9"),
        "--checkpoint".to_owned(),
        env_or("PETE_NEAT_CHECKPOINT", "data/models/locomotion_neat_v0"),
        "--report-dir".to_owned(),
        report_dir,
        "--state-checkpoint".to_owned(),
        path(&state)?.to_owned(),
        "--capture-root".to_owned(),
        env_or("PETE_NEAT_CAPTURE_ROOT", "data/captures/neat/locomotion-v2"),
        "--capture-every".to_owned(),
        env_or("PETE_NEAT_CAPTURE_EVERY", "2"),
        "--rehearsal-ratio".to_owned(),
        env_or("PETE_NEAT_REHEARSAL_RATIO", "0.20"),
        "--niche-audit-episodes".to_owned(),
        env_or("PETE_NEAT_NICHE_AUDIT_EPISODES", "16"),
        "--models-config".to_owned(),
        env_or("PETE_NEAT_MODELS_CONFIG", "configs/models.toml"),
    ];
    if env_flag("PETE_NEAT_NO_PROMOTE") {
        values.push("--no-promote".to_owned());
    }
    values.extend(extra);
    pete_tools(values.iter().map(String::as_str), &[])
}

fn neat_generation_limit(state: &Path, explicit: Option<u64>, default: u64, increment: u64) -> u64 {
    explicit
        .or_else(|| {
            fs::read_to_string(state)
                .ok()
                .and_then(|json| {
                    json.split("\"generation_in_stage\"")
                        .nth(1)?
                        .split(':')
                        .nth(1)?
                        .trim_start()
                        .split(|character: char| !character.is_ascii_digit())
                        .next()?
                        .parse::<u64>()
                        .ok()
                })
                .map(|completed| completed + increment)
        })
        .unwrap_or(default)
}

fn evolve(clear: Option<&str>, quality: bool) -> Result<()> {
    let prefix = if quality {
        "PETE_NEAT_QUALITY_"
    } else {
        "PETE_NEAT_"
    };
    let generations = env_or(
        &format!("{prefix}GENERATIONS"),
        if quality { "36" } else { "12" },
    );
    let population = env_or(
        &format!("{prefix}POPULATION"),
        if quality { "64" } else { "24" },
    );
    let hidden = env_or(
        &format!("{prefix}HIDDEN_DIM"),
        if quality { "14" } else { "10" },
    );
    let seed = env::var("PETE_NEAT_SEED").unwrap_or_else(|_| {
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
            .to_string()
    });
    let mut args = vec![
        "dream-train".to_owned(),
        "--start-level".to_owned(),
        env_or("PETE_NEAT_START_LEVEL", "motion"),
        "--generations".to_owned(),
        generations,
        "--population".to_owned(),
        population,
        "--seed".to_owned(),
        seed,
        "--hidden-dim".to_owned(),
        hidden,
        "--checkpoint-dir".to_owned(),
        env_or("PETE_NEAT_CHECKPOINT_DIR", "data/models/dream-policy/neat"),
        "--dataset-dir".to_owned(),
        env_or("PETE_NEAT_DATASET_DIR", "datasets/dream-policy/v0/episodes"),
        "--export-dataset".to_owned(),
        env_or("PETE_NEAT_EXPORT_DATASET", "false"),
    ];
    args.push("--detailed-logs".to_owned());
    if clear.is_some_and(|value| value == "true" || value == "--clear" || value == "clear=true") {
        args.push("--clear".to_owned());
    }
    println!(
        "Dream NEAT {}: {} generations, population {}, seed {}",
        if quality { "quality" } else { "fast" },
        args[4],
        args[6],
        args[8]
    );
    pete(["build", "--release", "-p", "pete-tools"])?;
    run_program(ProcessCommand::new("target/release/pete").args(args))
}

fn evolve_infinite(clear: Option<&str>) -> Result<()> {
    let dataset = PathBuf::from(env_or(
        "PETE_NEAT_DATASET_DIR",
        "datasets/dream-policy/v0/episodes",
    ));
    let export_dataset = env_flag("PETE_NEAT_EXPORT_DATASET");
    let checkpoint = PathBuf::from(env_or(
        "PETE_NEAT_CHECKPOINT_DIR",
        "data/models/dream-policy/neat",
    ))
    .join("evolve-best.json");
    let report_root = PathBuf::from(env_or(
        "PETE_EVOLVE_BENCHMARK_ROOT",
        "data/reports/scenario/evolve",
    ));
    let ledger_root = PathBuf::from(env_or(
        "PETE_EVOLVE_BENCHMARK_LEDGER_ROOT",
        "data/ledger/evolve-benchmark",
    ));
    let benchmark_every = env_u64("PETE_EVOLVE_BENCHMARK_EVERY", 10);
    let benchmark_steps = env_u64("PETE_EVOLVE_BENCHMARK_STEPS", 160).to_string();
    let max_runs = env_u64("PETE_EVOLVE_BENCHMARK_MAX_RUNS", 64) as usize;
    let benchmark_age = env_u64("PETE_EVOLVE_BENCHMARK_MAX_AGE_DAYS", 21);
    println!(
        "evolve-infinite: clear={} benchmark_every={benchmark_every} export_dataset={export_dataset}",
        clear.unwrap_or("false")
    );
    for iteration in 1_u64.. {
        println!("iteration #{iteration}");
        evolve(clear, true)?;
        if export_dataset {
            prune_dataset(&dataset)?;
        } else {
            println!("dataset: export disabled; skipping dataset retention");
        }
        if benchmark_every > 0 && iteration % benchmark_every == 0 {
            run_evolution_benchmarks(
                iteration,
                &benchmark_steps,
                &checkpoint,
                &report_root,
                &ledger_root,
            )?;
        }
        prune_directories(&report_root, max_runs, benchmark_age)?;
        prune_directories(&ledger_root, max_runs, benchmark_age)?;
        println!(
            "bench-retain: reports={}, ledgers={}",
            directory_count(&report_root),
            directory_count(&ledger_root)
        );
    }
    Ok(())
}

fn env_u64(name: &str, default: u64) -> u64 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn prune_dataset(root: &Path) -> Result<()> {
    fs::create_dir_all(root)?;
    let before = evolution_dataset_files(root)?;
    let max_age = Duration::from_secs(env_u64("PETE_DATASET_MAX_AGE_DAYS", 10) * 86_400);
    let max_files = env_u64("PETE_DATASET_MAX_FILES", 8_000) as usize;
    let max_bytes = env_u64("PETE_DATASET_MAX_BYTES", 536_870_912);
    let now = SystemTime::now();
    let mut retained = Vec::new();
    for (path, modified, size) in before.iter().cloned() {
        if !max_age.is_zero() && now.duration_since(modified).unwrap_or_default() > max_age {
            fs::remove_file(path)?;
        } else {
            retained.push((path, modified, size));
        }
    }
    retained.sort_by_key(|(_, modified, _)| *modified);
    while (max_files > 0 && retained.len() > max_files)
        || (max_bytes > 0 && retained.iter().map(|(_, _, size)| size).sum::<u64>() > max_bytes)
    {
        let (path, _, _) = retained.remove(0);
        fs::remove_file(path)?;
    }
    let size = retained.iter().map(|(_, _, size)| size).sum::<u64>();
    println!(
        "dataset: files {} -> {}, size={} bytes",
        before.len(),
        retained.len(),
        size
    );
    Ok(())
}

fn evolution_dataset_files(root: &Path) -> Result<Vec<(PathBuf, SystemTime, u64)>> {
    let mut files = Vec::new();
    collect_files(root, &mut files)?;
    Ok(files
        .into_iter()
        .filter_map(|path| {
            let name = path.file_name()?.to_string_lossy();
            if !name.starts_with("level-")
                || !name.contains("-seed-")
                || !name.contains("-genome-")
                || !name.ends_with(".jsonl")
            {
                return None;
            }
            let metadata = path.metadata().ok()?;
            Some((path, metadata.modified().ok()?, metadata.len()))
        })
        .collect())
}

fn collect_files(root: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    if !root.is_dir() {
        return Ok(());
    }
    for entry in fs::read_dir(root)? {
        let path = entry?.path();
        if path.is_dir() {
            collect_files(&path, files)?;
        } else if path.is_file() {
            files.push(path);
        }
    }
    Ok(())
}

fn run_evolution_benchmarks(
    iteration: u64,
    steps: &str,
    checkpoint: &Path,
    report_root: &Path,
    ledger_root: &Path,
) -> Result<()> {
    if !checkpoint.is_file() {
        println!("benchmark: skipped (missing {})", checkpoint.display());
        return Ok(());
    }
    let stamp = output("date", &["-u", "+%Y%m%dT%H%M%SZ"])?;
    let name = format!("{stamp}-iter-{iteration}");
    let reports = report_root.join(&name);
    let ledgers = ledger_root.join(name);
    fs::create_dir_all(&reports)?;
    fs::create_dir_all(&ledgers)?;
    for (scenario, seed) in [
        ("obstacle-avoidance", "701"),
        ("corner-trap", "1701"),
        ("column-trap", "2701"),
    ] {
        let ledger = ledgers.join(scenario);
        let report = reports.join(format!("{scenario}.json"));
        let _ = fs::remove_dir_all(&ledger);
        run_program(ProcessCommand::new("target/release/pete").args([
            "sim",
            "--scenario",
            scenario,
            "--steps",
            steps,
            "--tick-delay-ms",
            "0",
            "--seed",
            seed,
            "--ledger",
            path(&ledger)?,
            "--dream-policy-checkpoint",
            path(checkpoint)?,
        ]))?;
        run_program(ProcessCommand::new("target/release/pete").args([
            "virtual-report",
            "--ledger",
            path(&ledger)?,
            "--out",
            path(&report)?,
        ]))?;
    }
    println!("benchmark: reports at {}", reports.display());
    Ok(())
}

fn prune_directories(root: &Path, max_count: usize, max_age_days: u64) -> Result<()> {
    fs::create_dir_all(root)?;
    let now = SystemTime::now();
    let max_age = Duration::from_secs(max_age_days * 86_400);
    let mut directories = fs::read_dir(root)?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .filter_map(|path| Some((path.clone(), path.metadata().ok()?.modified().ok()?)))
        .collect::<Vec<_>>();
    for (path, modified) in &directories {
        if !max_age.is_zero() && now.duration_since(*modified).unwrap_or_default() > max_age {
            fs::remove_dir_all(path)?;
        }
    }
    directories.retain(|(path, _)| path.is_dir());
    directories.sort_by_key(|(_, modified)| *modified);
    if max_count > 0 && directories.len() > max_count {
        let remove = directories.len() - max_count;
        for (path, _) in directories.into_iter().take(remove) {
            fs::remove_dir_all(path)?;
        }
    }
    Ok(())
}

fn directory_count(root: &Path) -> usize {
    fs::read_dir(root)
        .map(|entries| {
            entries
                .flatten()
                .filter(|entry| entry.path().is_dir())
                .count()
        })
        .unwrap_or(0)
}

fn rehearse_models() -> Result<()> {
    for args in [
        ["sim", "--steps", "200", "--ledger", "data/ledger/sim1"].as_slice(),
        [
            "train",
            "behavior",
            "danger",
            "--ledger",
            "data/ledger/sim1",
        ]
        .as_slice(),
        [
            "train",
            "behavior",
            "charge",
            "--ledger",
            "data/ledger/sim1",
        ]
        .as_slice(),
        [
            "train",
            "behavior",
            "future",
            "--ledger",
            "data/ledger/sim1",
        ]
        .as_slice(),
        [
            "evaluate",
            "behavior",
            "danger",
            "--ledger",
            "data/ledger/sim1",
        ]
        .as_slice(),
        ["model-status"].as_slice(),
        [
            "sim",
            "--steps",
            "200",
            "--danger-checkpoint",
            "data/models/danger_v0",
            "--danger-mode",
            "shadow-infer",
        ]
        .as_slice(),
        [
            "robot",
            "--mode",
            "read-only",
            "--cockpit",
            "sim",
            "--steps",
            "20",
            "--capture",
            "data/captures/mock-readonly",
        ]
        .as_slice(),
        ["replay-capture", "--capture", "data/captures/mock-readonly"].as_slice(),
    ] {
        pete_tools(args.iter().copied(), &[])?;
    }
    Ok(())
}
