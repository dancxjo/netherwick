use chrono::{DateTime, Local};
use cliclack::{confirm, input, intro, log, multiselect, note, outro, outro_cancel, select};
use std::{
    env, fs,
    io::{self, IsTerminal},
    path::PathBuf,
    process::Command as ProcessCommand,
};

type Result<T> = super::Result<T>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum QaCase {
    Charging,
    BumperLeft,
    BumperRight,
    CliffLeft,
    CliffFrontLeft,
    CliffFrontRight,
    CliffRight,
    WheelDrop,
    HeartbeatLoss,
    TransportLoss,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Outcome {
    Pass,
    Fail,
    Blocked,
    Skip,
}

struct CaseDefinition {
    case: QaCase,
    title: &'static str,
    hint: &'static str,
    setup: &'static str,
    procedure: &'static str,
    acceptance: &'static str,
    guarded_bumper_helper: bool,
}

struct CaseResult {
    definition: &'static CaseDefinition,
    outcome: Outcome,
    notes: String,
    helper_succeeded: Option<bool>,
}

static CASES: &[CaseDefinition] = &[
    CaseDefinition {
        case: QaCase::Charging,
        title: "Charging interlock",
        hint: "queued, autonomous, and direct motion stay blocked",
        setup: "Stage Pete on the dock or otherwise make the real charging indicator active. Keep an emergency STOP within reach.",
        procedure: "In a second terminal run `just possess --autonomous-motion --dashboard 127.0.0.1:8787 --duration-seconds 45`. While charging remains active, request Forward from the dashboard and allow executive ticks to propose motion. Inspect the runner output and capture for explicit charging-gate refusals.",
        acceptance: "No wheel motion occurs for queued, autonomous, or direct requests; each refusal is attributed to the charging interlock; shutdown ends stopped and exorcized.",
        guarded_bumper_helper: false,
    },
    CaseDefinition {
        case: QaCase::BumperLeft,
        title: "Left bumper recovery",
        hint: "contact, stop, reverse, turn right, probe, inspect",
        setup: "Lift and securely support the drive wheels. Keep hands, hair, and cables clear of the wheels.",
        procedure: "The runner can launch the guarded recovery helper. When it asks for contact, press and hold only the LEFT bumper, then release it when instructed.",
        acceptance: "The helper verifies contact -> stop -> clear -> reverse -> turn right -> probe -> inspect, then reports stopped and exorcized.",
        guarded_bumper_helper: true,
    },
    CaseDefinition {
        case: QaCase::BumperRight,
        title: "Right bumper recovery",
        hint: "contact, stop, reverse, turn left, probe, inspect",
        setup: "Lift and securely support the drive wheels. Keep hands, hair, and cables clear of the wheels.",
        procedure: "The runner can launch the guarded recovery helper. When it asks for contact, press and hold only the RIGHT bumper, then release it when instructed.",
        acceptance: "The helper verifies contact -> stop -> clear -> reverse -> turn left -> probe -> inspect, then reports stopped and exorcized.",
        guarded_bumper_helper: true,
    },
    CaseDefinition {
        case: QaCase::CliffLeft,
        title: "Left cliff sensor",
        hint: "active motion stops and remains vetoed",
        setup: "Support Pete so the selected cliff sensor can be exposed without allowing a fall. Begin with every cliff sensor clear.",
        procedure: "In a second terminal run `just possess --dashboard 127.0.0.1:8787 --duration-seconds 90`, open `http://127.0.0.1:8787/view`, request a slow Forward pulse, then expose only the LEFT cliff sensor to the test edge. Preserve the safety and motion events from the capture.",
        acceptance: "The left cliff flag appears, active motion stops, further motion remains vetoed while active, and shutdown ends stopped and exorcized.",
        guarded_bumper_helper: false,
    },
    CaseDefinition {
        case: QaCase::CliffFrontLeft,
        title: "Front-left cliff sensor",
        hint: "active motion stops and remains vetoed",
        setup: "Support Pete so the selected cliff sensor can be exposed without allowing a fall. Begin with every cliff sensor clear.",
        procedure: "In a second terminal run `just possess --dashboard 127.0.0.1:8787 --duration-seconds 90`, open `http://127.0.0.1:8787/view`, request a slow Forward pulse, then expose only the FRONT-LEFT cliff sensor to the test edge. Preserve the safety and motion events from the capture.",
        acceptance: "The front-left cliff flag appears, active motion stops, further motion remains vetoed while active, and shutdown ends stopped and exorcized.",
        guarded_bumper_helper: false,
    },
    CaseDefinition {
        case: QaCase::CliffFrontRight,
        title: "Front-right cliff sensor",
        hint: "active motion stops and remains vetoed",
        setup: "Support Pete so the selected cliff sensor can be exposed without allowing a fall. Begin with every cliff sensor clear.",
        procedure: "In a second terminal run `just possess --dashboard 127.0.0.1:8787 --duration-seconds 90`, open `http://127.0.0.1:8787/view`, request a slow Forward pulse, then expose only the FRONT-RIGHT cliff sensor to the test edge. Preserve the safety and motion events from the capture.",
        acceptance: "The front-right cliff flag appears, active motion stops, further motion remains vetoed while active, and shutdown ends stopped and exorcized.",
        guarded_bumper_helper: false,
    },
    CaseDefinition {
        case: QaCase::CliffRight,
        title: "Right cliff sensor",
        hint: "active motion stops and remains vetoed",
        setup: "Support Pete so the selected cliff sensor can be exposed without allowing a fall. Begin with every cliff sensor clear.",
        procedure: "In a second terminal run `just possess --dashboard 127.0.0.1:8787 --duration-seconds 90`, open `http://127.0.0.1:8787/view`, request a slow Forward pulse, then expose only the RIGHT cliff sensor to the test edge. Preserve the safety and motion events from the capture.",
        acceptance: "The right cliff flag appears, active motion stops, further motion remains vetoed while active, and shutdown ends stopped and exorcized.",
        guarded_bumper_helper: false,
    },
    CaseDefinition {
        case: QaCase::WheelDrop,
        title: "Wheel-drop latch",
        hint: "stop stays latched and overrides recovery",
        setup: "Securely support Pete before lifting a wheel. Keep the opposite wheel from driving the body off its support.",
        procedure: "In a second terminal run `just possess --dashboard 127.0.0.1:8787 --duration-seconds 90` and use `http://127.0.0.1:8787/view` to request bounded Forward motion. Lift one wheel until wheel-drop telemetry is active. Attempt Forward and recovery while the condition remains active, then lower the wheel and inspect the latch before manually clearing it.",
        acceptance: "Motion stops, wheel-drop is identified, direct and recovery motion remain vetoed, and the latch does not clear merely because the wheel is lowered.",
        guarded_bumper_helper: false,
    },
    CaseDefinition {
        case: QaCase::HeartbeatLoss,
        title: "Heartbeat loss",
        hint: "firmware stops motion without a host heartbeat",
        setup: "Lift and securely support the drive wheels. Start bounded possession with visible wheel motion and keep emergency STOP available.",
        procedure: "In a second terminal run `just possess --dashboard 127.0.0.1:8787 --duration-seconds 90`, use `http://127.0.0.1:8787/view` to request bounded Forward motion, then suspend the possession process (`Ctrl-Z`) so no heartbeat reaches the brainstem. Observe the wheels, wait at least one second, resume with `fg`, and inspect events before exiting normally.",
        acceptance: "The brainstem emits `HeartbeatExpired` at its 750 ms deadline and motion is stopped without a host STOP; normal shutdown still ends stopped and exorcized after resume.",
        guarded_bumper_helper: false,
    },
    CaseDefinition {
        case: QaCase::TransportLoss,
        title: "Transport loss and reconnect",
        hint: "motion stops; reconnect waits for fresh packet 0",
        setup: "Lift and securely support the drive wheels. Use the exact configured UART or Wi-Fi transport and keep a way to restore it immediately.",
        procedure: "In a second terminal run `just possess --dashboard 127.0.0.1:8787 --duration-seconds 120`, use `http://127.0.0.1:8787/view` to request bounded Forward motion, then interrupt only the selected Cockpit transport. Confirm physical stop, restore the same transport, and watch the runner retry. Preserve the disconnect and reconnect log plus capture manifest.",
        acceptance: "Motion stops on transport loss; the old session/lease is not reused; reconnect begins stopped and opens the runner only after the complete packet-0 counter advances with age <= 500 ms.",
        guarded_bumper_helper: false,
    },
];

pub fn run(
    plan: bool,
    out: Option<PathBuf>,
    run_possession: fn(&[String]) -> Result<()>,
) -> Result<()> {
    if plan {
        print!("{}", render_plan());
        return Ok(());
    }
    if !io::stdin().is_terminal() || !io::stderr().is_terminal() {
        return Err(
            "physical QA is interactive; run it in a terminal or use `just physical-qa --plan`"
                .into(),
        );
    }

    intro("🧰 Pete physical QA")?;
    note(
        "Safety first",
        "This runner never treats a prompt as a safety gate. Use the physical STOP, secure the robot for each case, and abort any motion that does not match the instructions.",
    )?;
    if !confirm("Is Pete in a controlled test area with emergency STOP reachable?")
        .initial_value(false)
        .interact()?
    {
        outro_cancel("QA cancelled before touching hardware")?;
        return Ok(());
    }

    let operator_default = env::var("USER").unwrap_or_else(|_| "operator".to_owned());
    let operator: String = input("Who is running this session?")
        .default_input(&operator_default)
        .interact()?;
    let session_notes: String = input("Session label or setup note")
        .placeholder("bench, floor rig, firmware change under test")
        .required(false)
        .interact()?;

    let firmware = read_firmware_identity();
    match &firmware {
        Ok(identity) => note("Brainstem firmware identity", identity)?,
        Err(error) => log::warning(format!("Could not read /status.json: {error}"))?,
    }
    if firmware.is_err()
        && !confirm("Continue without an automatically captured firmware identity?")
            .initial_value(false)
            .interact()?
    {
        outro_cancel("QA cancelled; connect to the brainstem and try again")?;
        return Ok(());
    }

    let all_cases: Vec<QaCase> = CASES.iter().map(|definition| definition.case).collect();
    let items: Vec<_> = CASES
        .iter()
        .map(|definition| (definition.case, definition.title, definition.hint))
        .collect();
    let selected = multiselect("Which cases should this session run?")
        .items(&items)
        .initial_values(all_cases)
        .max_rows(10)
        .interact()?;

    let mut results = Vec::new();
    for definition in CASES {
        if !selected.contains(&definition.case) {
            continue;
        }
        log::step(definition.title)?;
        note("Set up", definition.setup)?;
        note("Procedure", definition.procedure)?;
        note("Pass when", definition.acceptance)?;

        let helper_succeeded = if definition.guarded_bumper_helper
            && confirm("Launch the guarded bumper-recovery helper now?")
                .initial_value(true)
                .interact()?
        {
            log::info(
                "Handing the terminal to `just possess --recovery-smoke --wheels-off-floor`",
            )?;
            let args = vec![
                "--recovery-smoke".to_owned(),
                "--wheels-off-floor".to_owned(),
            ];
            let succeeded = match run_possession(&args) {
                Ok(()) => {
                    log::success("Guarded helper completed")?;
                    true
                }
                Err(error) => {
                    log::error(format!("Guarded helper failed: {error}"))?;
                    false
                }
            };
            Some(succeeded)
        } else {
            None
        };

        let initial = if helper_succeeded == Some(true) {
            Outcome::Pass
        } else if helper_succeeded == Some(false) {
            Outcome::Fail
        } else {
            Outcome::Pass
        };
        let outcome = select("Record this case")
            .item(Outcome::Pass, "Pass", "acceptance evidence was observed")
            .item(
                Outcome::Fail,
                "Fail",
                "behavior disagreed with the acceptance gate",
            )
            .item(
                Outcome::Blocked,
                "Blocked",
                "setup or hardware prevented a verdict",
            )
            .item(Outcome::Skip, "Skip", "leave the case pending")
            .initial_value(initial)
            .interact()?;
        let notes: String = input("Evidence note")
            .placeholder("capture path, event sequence, timing, failure details")
            .required(outcome != Outcome::Skip)
            .interact()?;
        results.push(CaseResult {
            definition,
            outcome,
            notes,
            helper_succeeded,
        });
    }

    let now = Local::now();
    let output = out.unwrap_or_else(|| default_report_path(now));
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let report = render_report(
        now,
        &operator,
        &session_notes,
        firmware.as_deref().unwrap_or("unavailable"),
        &selected,
        &results,
    );
    fs::write(&output, report)?;

    let passed = results
        .iter()
        .filter(|result| result.outcome == Outcome::Pass)
        .count();
    let unresolved = results.len().saturating_sub(passed);
    outro(format!(
        "Recorded {passed} passed and {unresolved} unresolved cases in {}",
        output.display()
    ))?;
    Ok(())
}

fn read_firmware_identity() -> std::result::Result<String, String> {
    let host = env::var("PETE_BRAINSTEM_HTTP_HOST").unwrap_or_else(|_| "192.168.4.1:80".to_owned());
    let base = if host.starts_with("http://") || host.starts_with("https://") {
        host.trim_end_matches('/').to_owned()
    } else {
        format!("http://{}", host.trim_end_matches('/'))
    };
    let output = ProcessCommand::new("curl")
        .args(["-fsS", "--max-time", "3", &format!("{base}/status.json")])
        .output()
        .map_err(|error| format!("could not start curl: {error}"))?;
    if !output.status.success() {
        return Err(format!("curl exited with {}", output.status));
    }
    let body = String::from_utf8(output.stdout)
        .map_err(|error| format!("status response was not UTF-8: {error}"))?;
    select_firmware_identity(&body)
}

fn select_firmware_identity(body: &str) -> std::result::Result<String, String> {
    let status: serde_json::Value = serde_json::from_str(body.trim())
        .map_err(|error| format!("status response was not valid JSON: {error}"))?;
    let status = status
        .as_object()
        .ok_or_else(|| "status response was not a JSON object".to_owned())?;
    let mut identity = serde_json::Map::new();
    for key in [
        "firmware_name",
        "firmware_version",
        "git_commit",
        "git_commit_short",
        "git_dirty",
        "build_timestamp",
        "build_profile",
        "build_target",
        "build_backend",
        "build_id",
    ] {
        if let Some(value) = status.get(key) {
            identity.insert(key.to_owned(), value.clone());
        }
    }
    if identity.is_empty() {
        return Err("status response did not include firmware identity fields".to_owned());
    }
    serde_json::to_string_pretty(&identity)
        .map_err(|error| format!("could not format firmware identity: {error}"))
}

fn default_report_path(now: DateTime<Local>) -> PathBuf {
    PathBuf::from(format!(
        "data/reports/physical-qa/{}.md",
        now.format("%Y%m%d-%H%M%S")
    ))
}

fn render_plan() -> String {
    let mut out = String::from("# Pete physical QA plan\n\n");
    out.push_str("Run interactively with `just physical-qa`. The runner records firmware identity, operator notes, outcomes, and evidence in a timestamped Markdown report.\n\n");
    for (index, definition) in CASES.iter().enumerate() {
        out.push_str(&format!("{}. {}\n", index + 1, definition.title));
        out.push_str(&format!("   Set up: {}\n", definition.setup));
        out.push_str(&format!("   Procedure: {}\n", definition.procedure));
        out.push_str(&format!("   Pass when: {}\n", definition.acceptance));
    }
    out
}

fn render_report(
    now: DateTime<Local>,
    operator: &str,
    session_notes: &str,
    firmware: &str,
    selected: &[QaCase],
    results: &[CaseResult],
) -> String {
    let mut out = format!(
        "# Physical QA session\n\n- Started: {}\n- Operator: {}\n- Session: {}\n\n## Brainstem firmware identity\n\n```json\n{}\n```\n\n## Results\n\n",
        now.to_rfc3339(),
        markdown_text(operator),
        markdown_text(if session_notes.is_empty() { "not supplied" } else { session_notes }),
        firmware.replace("```", "` ` `"),
    );
    for definition in CASES {
        let result = results
            .iter()
            .find(|result| result.definition.case == definition.case);
        let (marker, outcome, notes, helper) = match result {
            Some(result) => (
                if result.outcome == Outcome::Pass {
                    "x"
                } else {
                    " "
                },
                outcome_label(result.outcome),
                if result.notes.is_empty() {
                    "none"
                } else {
                    &result.notes
                },
                match result.helper_succeeded {
                    Some(true) => "guarded helper succeeded",
                    Some(false) => "guarded helper failed",
                    None => "not run",
                },
            ),
            None if selected.contains(&definition.case) => (" ", "not recorded", "none", "not run"),
            None => (" ", "not selected", "none", "not run"),
        };
        out.push_str(&format!(
            "- [{marker}] **{}** — {outcome}; helper: {helper}; evidence: {}\n",
            definition.title,
            markdown_text(notes)
        ));
    }
    out.push_str("\n## Acceptance gates\n\n");
    for definition in CASES {
        out.push_str(&format!(
            "- **{}:** {}\n",
            definition.title, definition.acceptance
        ));
    }
    out
}

fn outcome_label(outcome: Outcome) -> &'static str {
    match outcome {
        Outcome::Pass => "pass",
        Outcome::Fail => "fail",
        Outcome::Blocked => "blocked",
        Outcome::Skip => "skipped",
    }
}

fn markdown_text(value: &str) -> String {
    value.replace('\n', " ").replace('|', "\\|")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_names_every_physical_case_and_operator_entrypoint() {
        let plan = render_plan();
        assert!(plan.contains("just physical-qa"));
        for definition in CASES {
            assert!(plan.contains(definition.title));
            assert!(plan.contains(definition.acceptance));
        }
    }

    #[test]
    fn report_keeps_unselected_and_failed_cases_pending() {
        let result = CaseResult {
            definition: &CASES[0],
            outcome: Outcome::Fail,
            notes: "wheels moved".to_owned(),
            helper_succeeded: None,
        };
        let report = render_report(
            Local::now(),
            "Pete Operator",
            "bench",
            r#"{"git_commit":"abc"}"#,
            &[QaCase::Charging],
            &[result],
        );
        assert!(report.contains("- [ ] **Charging interlock** — fail"));
        assert!(report.contains("wheels moved"));
        assert!(report.contains("- [ ] **Left bumper recovery** — not selected"));
    }

    #[test]
    fn firmware_identity_omits_unrelated_live_status() {
        let identity = select_firmware_identity(
            r#"{"firmware_name":"pete","git_commit":"abc","active_motion":true}"#,
        )
        .unwrap();
        assert!(identity.contains("firmware_name"));
        assert!(identity.contains("git_commit"));
        assert!(!identity.contains("active_motion"));
    }
}
