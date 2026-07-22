#[derive(Clone, Debug, Parser)]
struct ShadowScoreArgs {
    /// Completed shadow-flight artifact directory.
    #[arg(long)]
    run: PathBuf,
    /// An existing certification JSON report to compare as the baseline.
    #[arg(long)]
    baseline: Option<PathBuf>,
    /// Output directory. Defaults to the run directory.
    #[arg(long)]
    output: Option<PathBuf>,
    /// Replace this baseline path with the current report.
    #[arg(long)]
    update_baseline: Option<PathBuf>,
    /// Required acknowledgement for any baseline replacement.
    #[arg(long)]
    reviewed_baseline_update: bool,
}

fn run_shadow_score(args: ShadowScoreArgs) -> Result<()> {
    let output = args.output.as_deref().unwrap_or(&args.run);
    fs::create_dir_all(output)?;
    let manifest: ShadowFlightManifest = serde_json::from_slice(&fs::read(args.run.join("manifest.json"))?)?;
    let events = fs::read_to_string(args.run.join(&manifest.events_path))?
        .lines()
        .map(serde_json::from_str::<ShadowEventRecord>)
        .collect::<std::result::Result<Vec<_>, _>>()?
        .into_iter()
        .map(|record| record.event)
        .collect::<Vec<_>>();
    let software_artifact = current_git_commit()
        .map(|commit| format!("git:{commit}"))
        .unwrap_or_else(|| "git:unknown".into());
    let config_artifact = format!(
        "sha256:{:x}",
        Sha256::digest(serde_json::to_vec(&serde_json::json!({
            "source": manifest.source,
            "seed": manifest.seed,
            "clock_mode": manifest.clock_mode,
            "higher_brain_mode": manifest.higher_brain_mode,
            "tick_ms": manifest.tick_ms,
            "production_components": manifest.production_components,
            "substitutions": manifest.substitutions,
            "ledger_retained_frames": manifest.ledger_retained_frames,
            "ledger_retained_transitions": manifest.ledger_retained_transitions,
            "event_retention_limit": manifest.event_retention_limit,
            "input_retention_limit": manifest.input_retention_limit,
        }))?)
    );
    let identity = pete_worldlab::CertificationRunIdentity::deterministic(
        manifest.input_frames_sha256.clone(),
        software_artifact,
        config_artifact,
        manifest.source_identity.clone(),
        manifest.seed,
    );
    let bundle_references = vec![format!("shadow-flight://{}", args.run.display())];
    let report = pete_worldlab::score_shadow_events(identity, &events, &bundle_references);
    fs::write(
        output.join("certification.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    fs::write(
        output.join("observatory-certification.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "schema_version": 1,
            "kind": "shadow_flight_certification",
            "report": report,
        }))?,
    )?;

    if let Some(baseline_path) = args.baseline {
        let baseline: pete_worldlab::ShadowCertificationReport =
            serde_json::from_slice(&fs::read(&baseline_path)?)?;
        let comparison = pete_worldlab::compare_certification_reports(&baseline, &report);
        fs::write(
            output.join("comparison.json"),
            serde_json::to_vec_pretty(&comparison)?,
        )?;
        println!(
            "comparison: comparable={} regressions={} baseline={} candidate={}",
            comparison.comparable,
            comparison.regressions.len(),
            comparison.baseline_run_id,
            comparison.candidate_run_id
        );
        if !comparison.comparable {
            anyhow::bail!(
                "baseline and candidate are not comparable: {}",
                comparison.incompatibilities.join(", ")
            );
        }
    }

    if let Some(baseline_path) = args.update_baseline {
        let bytes = pete_worldlab::authorize_baseline_update(
            &report,
            args.reviewed_baseline_update,
        )
        .map_err(anyhow::Error::msg)?;
        if let Some(parent) = baseline_path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(&baseline_path, bytes)?;
        println!("reviewed baseline updated: {}", baseline_path.display());
    } else if args.reviewed_baseline_update {
        anyhow::bail!("--reviewed-baseline-update requires --update-baseline");
    }

    println!(
        "certification: {} gates={} metrics={} report={}",
        if report.passed { "PASS" } else { "FAIL" },
        report.gates.len(),
        report.metrics.len(),
        output.join("certification.json").display()
    );
    for gate in report.gates.iter().filter(|gate| !gate.passed) {
        println!(
            "  failed invariant {}: {} (events: {})",
            gate.invariant,
            gate.failure_reason.as_deref().unwrap_or("threshold failed"),
            gate.supporting_event_ids.join(",")
        );
    }
    if !report.passed {
        anyhow::bail!("shadow-flight certification gates failed");
    }
    Ok(())
}
