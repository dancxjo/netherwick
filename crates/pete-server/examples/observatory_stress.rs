use std::path::PathBuf;

use pete_server::{run_observatory_stress, ObservatoryStressConfig};

fn argument(name: &str) -> Option<String> {
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        if argument == name {
            return args.next();
        }
    }
    None
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let profile = argument("--profile").unwrap_or_else(|| "ci".into());
    let output = PathBuf::from(
        argument("--output")
            .unwrap_or_else(|| "data/reports/observatory-stress/software.json".into()),
    );
    let durable_directory = argument("--durable-dir")
        .map(PathBuf::from)
        .unwrap_or_else(|| output.with_extension("data"));
    let mut config = if profile == "pi5-soak" {
        ObservatoryStressConfig::pi5_soak(durable_directory)
    } else if profile == "ci" {
        ObservatoryStressConfig::ci(durable_directory)
    } else {
        return Err(format!("unknown profile {profile:?}; expected ci or pi5-soak").into());
    };
    if let Some(events) = argument("--events") {
        config.events = events.parse()?;
    }
    if config.events < 10_001 {
        return Err("--events must be at least 10001 to exercise a clock reset".into());
    }
    let report = run_observatory_stress(config).await?;
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
    println!(
        "Observatory stress {}: {} events, publish p99 added {:.3} us, RSS growth {:?} bytes, report {}",
        if report.passed { "PASS" } else { "FAIL" },
        report.metrics.attempted_events,
        report.metrics.added_publish_p99_us,
        report.metrics.rss_growth_bytes,
        output.display()
    );
    if !report.passed {
        std::process::exit(2);
    }
    Ok(())
}
