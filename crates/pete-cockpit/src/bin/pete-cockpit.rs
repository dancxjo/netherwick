use std::error::Error;
use std::thread;
use std::time::{Duration, Instant};

use clap::{Parser, Subcommand, ValueEnum};
use pete_cockpit::{
    discover_usb_serial_by_id, establish_session, Cockpit, CockpitError, CockpitRequest,
    CockpitResponse, ControlAuthority, DockIrCue, HandshakeHello, HttpCockpit, SessionCockpit,
    SessionPurpose, StatusSummary, UartCockpit, UartCockpitConfig,
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
type AnySession = SessionCockpit<Box<dyn Cockpit>>;

#[derive(Debug, Parser)]
#[command(
    name = "pete-cockpit",
    about = "Transport-neutral operator cockpit for Pete's Brainstem"
)]
struct Cli {
    #[arg(long, global = true, value_enum, default_value = "usb")]
    transport: Transport,
    /// USB/UART path or HTTP host:port. USB auto-discovers when omitted.
    #[arg(long, global = true)]
    endpoint: Option<String>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum Transport {
    /// USB CDC, auto-discovered under /dev/serial/by-id by default.
    Usb,
    /// An explicit serial UART or USB CDC path.
    Uart,
    /// Brainstem HTTP cockpit.
    Http,
}

#[derive(Debug, Subcommand)]
enum Command {
    Status,
    Capabilities,
    Events {
        #[arg(long)]
        since: Option<u32>,
    },
    Stop,
    /// Configure Brainstem-owned auditory annunciation.
    Audio {
        #[command(subcommand)]
        mode: AudioCommand,
    },
    /// Renew a bounded velocity primitive while live safety telemetry stays healthy.
    Drive {
        #[arg(long, allow_hyphen_values = true)]
        linear: i16,
        #[arg(long, default_value_t = 0, allow_hyphen_values = true)]
        angular: i16,
        #[arg(long, default_value_t = 500)]
        duration_ms: u64,
        /// Explicitly renew CAREFUL while driving, making sensor gates advisory.
        #[arg(long)]
        careful: bool,
    },
    /// Follow the Create Home Base IR gradient with short-lived primitives.
    DockAlign {
        #[arg(long, default_value_t = 4_000)]
        duration_ms: u64,
        #[arg(long, default_value_t = 120)]
        speed: i16,
        #[arg(long, default_value_t = 400)]
        correction: i16,
        /// Explicitly renew CAREFUL while aligning.
        #[arg(long)]
        careful: bool,
    },
}

#[derive(Debug, Subcommand)]
enum AudioCommand {
    Silent,
    Audible,
}

fn main() -> Result<()> {
    let _ = dotenvy::dotenv();
    let cli = Cli::parse();
    let mut session = connect(&cli)?;
    match cli.command {
        Command::Status => print_json(&read_status(&mut session)?),
        Command::Capabilities => {
            let capabilities = match session.execute(CockpitRequest::GetCapabilities)? {
                CockpitResponse::Capabilities(capabilities) => capabilities,
                response => {
                    return Err(format!("unexpected capabilities response: {response:?}").into());
                }
            };
            print_json(&capabilities)
        }
        Command::Events { since } => {
            let since = since.unwrap_or_else(|| session.event_cursor_hint().unwrap_or(0));
            let events = match session.execute(CockpitRequest::GetEvents { since_seq: since })? {
                CockpitResponse::Events(events) => events,
                response => return Err(format!("unexpected events response: {response:?}").into()),
            };
            print_json(&events)
        }
        Command::Stop => {
            accepted(session.execute(CockpitRequest::Stop)?)?;
            println!("stopped");
            Ok(())
        }
        Command::Audio { mode } => {
            let silent = matches!(mode, AudioCommand::Silent);
            accepted(session.execute(CockpitRequest::SetAudioSilent { silent })?)?;
            let status = read_status(&mut session)?;
            if status.audio_silent != Some(silent) {
                return Err("Brainstem did not confirm the requested audio state".into());
            }
            println!("audio {}", if silent { "silent" } else { "audible" });
            Ok(())
        }
        Command::Drive {
            linear,
            angular,
            duration_ms,
            careful,
        } => run_guarded_motion(&mut session, duration_ms, careful, |status| {
            let _ = status;
            Ok(Some((linear, angular, "bounded drive")))
        }),
        Command::DockAlign {
            duration_ms,
            speed,
            correction,
            careful,
        } => run_guarded_motion(&mut session, duration_ms, careful, |status| {
            if status.battery.home_base()
                || status.battery.charging_indicator == Some(true)
                || status.battery.charging_state.unwrap_or(0) != 0
            {
                return Ok(None);
            }
            let cue = DockIrCue::from_character(status.infrared_character.unwrap_or(0))
                .ok_or("Home Base IR gradient disappeared")?;
            Ok(Some((
                speed,
                cue.steering_mrad_s(correction),
                "Home Base IR gradient",
            )))
        }),
    }
}

fn connect(cli: &Cli) -> Result<AnySession> {
    let hello = operator_control_hello();
    match cli.transport {
        Transport::Http => {
            let endpoint = cli.endpoint.as_deref().unwrap_or("192.168.4.1:80");
            establish(Box::new(HttpCockpit::connect(endpoint)), hello)
        }
        Transport::Uart => {
            let endpoint = cli
                .endpoint
                .as_deref()
                .ok_or("--transport uart requires --endpoint PATH")?;
            establish(Box::new(open_serial(endpoint)?), hello)
        }
        Transport::Usb => {
            if let Some(endpoint) = cli.endpoint.as_deref() {
                return establish(Box::new(open_serial(endpoint)?), hello);
            }
            let paths = discover_usb_serial_by_id()?;
            if paths.is_empty() {
                return Err("no Brainstem USB CDC endpoint found under /dev/serial/by-id".into());
            }
            let mut failures = Vec::new();
            for path in paths {
                let path_text = path.to_string_lossy();
                match open_serial(path_text.as_ref())
                    .map(|cockpit| Box::new(cockpit) as Box<dyn Cockpit>)
                    .and_then(|cockpit| {
                        thread::sleep(Duration::from_millis(250));
                        establish_session(cockpit, hello.new_attempt(), None)
                    }) {
                    Ok(session) => return Ok(session),
                    Err(error) => failures.push(format!("{}: {error}", path.display())),
                }
            }
            Err(format!(
                "all Brainstem USB candidates failed: {}",
                failures.join("; ")
            )
            .into())
        }
    }
}

fn operator_control_hello() -> HandshakeHello {
    let mut hello = HandshakeHello::operator("pete-cockpit-cli");
    hello.session_purpose = SessionPurpose::Control;
    hello
}

fn open_serial(endpoint: &str) -> pete_cockpit::Result<UartCockpit> {
    UartCockpit::connect_with_config(
        UartCockpitConfig::new(endpoint)
            .with_timeout(Duration::from_secs(2))
            .with_data_terminal_ready(true),
    )
}

fn establish(connector: Box<dyn Cockpit>, hello: HandshakeHello) -> Result<AnySession> {
    thread::sleep(Duration::from_millis(250));
    Ok(establish_session(connector, hello, None)?)
}

fn run_guarded_motion<F>(
    session: &mut AnySession,
    duration_ms: u64,
    careful: bool,
    mut command: F,
) -> Result<()>
where
    F: FnMut(&StatusSummary) -> Result<Option<(i16, i16, &'static str)>>,
{
    if duration_ms == 0 || duration_ms > 30_000 {
        return Err("--duration-ms must be between 1 and 30000".into());
    }
    session.acquire_control(ControlAuthority::OperatorDebug, 60_000)?;
    accepted(session.execute(CockpitRequest::Stop)?)?;
    execute_with_busy_retry(
        session,
        CockpitRequest::StreamSensors {
            enabled: true,
            packet_id: 0,
            period_ms: 100,
        },
    )?;
    wait_ready(session)?;

    let started = Instant::now();
    let result = (|| -> Result<()> {
        while started.elapsed() < Duration::from_millis(duration_ms) {
            if careful {
                accepted(session.execute(CockpitRequest::CarefulMode { ttl_ms: 1_000 })?)?;
            }
            let status = read_status(session)?;
            let Some((linear_mm_s, angular_mrad_s, reason)) = command(&status)? else {
                println!(
                    "complete elapsed_ms={} ir={:?} home_base={} charging={} safety_latch={:?}",
                    started.elapsed().as_millis(),
                    status.infrared_character,
                    status.battery.home_base(),
                    status.battery.charging_state.unwrap_or(0) != 0
                        || status.battery.charging_indicator == Some(true),
                    status.safety_latch_kind,
                );
                return Ok(());
            };
            guard(&status)?;
            if linear_mm_s.abs() > 500 || angular_mrad_s.abs() > 2_000 {
                return Err("motion exceeds cockpit CLI bounds (500 mm/s, 2000 mrad/s)".into());
            }
            accepted(session.execute(CockpitRequest::CmdVel {
                linear_mm_s,
                angular_mrad_s,
                ttl_ms: 250,
            })?)?;
            println!(
                "motion elapsed_ms={} linear={} angular={} ir={:?} cue={reason}",
                started.elapsed().as_millis(),
                linear_mm_s,
                angular_mrad_s,
                status.infrared_character
            );
            thread::sleep(Duration::from_millis(80));
        }
        println!(
            "complete elapsed_ms={} reason=duration",
            started.elapsed().as_millis()
        );
        Ok(())
    })();
    let stopped = accepted(session.execute(CockpitRequest::Stop)?);
    result?;
    stopped
}

fn wait_ready(session: &mut AnySession) -> Result<()> {
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        let status = read_status(session)?;
        if status.has_fresh_complete_body_packet(300) {
            return guard(&status);
        }
        if Instant::now() >= deadline {
            return Err("no fresh complete Create packet 0 arrived".into());
        }
        thread::sleep(Duration::from_millis(50));
    }
}

fn guard(status: &StatusSummary) -> Result<()> {
    if status.safety_tripped == Some(true) {
        return Err(format!(
            "Brainstem safety is tripped: {:?}",
            status.safety_latch_kind
        )
        .into());
    }
    if status.estop_latched == Some(true) {
        return Err("Brainstem e-stop is latched".into());
    }
    if !status.has_fresh_complete_body_packet(300) {
        return Err("Create packet 0 telemetry became stale".into());
    }
    Ok(())
}

fn read_status(session: &mut AnySession) -> Result<StatusSummary> {
    match session.execute(CockpitRequest::GetStatus)? {
        CockpitResponse::Status(status) => Ok(status.summary()),
        response => Err(format!("unexpected status response: {response:?}").into()),
    }
}

fn execute_with_busy_retry(session: &mut AnySession, request: CockpitRequest) -> Result<()> {
    for attempt in 0..=5 {
        match session.execute(request.clone()) {
            Ok(response) => return accepted(response),
            Err(CockpitError::Rejected { reason, .. }) if reason == "busy" && attempt < 5 => {
                thread::sleep(Duration::from_millis(180));
            }
            Err(error) => return Err(error.into()),
        }
    }
    unreachable!("bounded retry returns on its final attempt")
}

fn accepted(response: CockpitResponse) -> Result<()> {
    match response {
        CockpitResponse::Accepted => Ok(()),
        response => Err(format!("command was not accepted: {response:?}").into()),
    }
}

fn print_json(value: &impl serde::Serialize) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{operator_control_hello, AudioCommand, Cli, Command};
    use clap::Parser;
    use pete_cockpit::SessionPurpose;

    #[test]
    fn motion_cli_accepts_a_separate_negative_velocity_argument() {
        let cli = Cli::try_parse_from([
            "pete-cockpit",
            "drive",
            "--linear",
            "-120",
            "--duration-ms",
            "2500",
        ])
        .unwrap();

        assert!(matches!(cli.command, Command::Drive { linear: -120, .. }));
    }

    #[test]
    fn operator_motion_handshake_requests_control_purpose() {
        assert_eq!(
            operator_control_hello().session_purpose,
            SessionPurpose::Control
        );
    }

    #[test]
    fn audio_cli_accepts_silent_and_audible_modes() {
        for (mode, silent) in [("silent", true), ("audible", false)] {
            let cli = Cli::try_parse_from(["pete-cockpit", "audio", mode]).unwrap();
            assert!(
                matches!(
                    cli.command,
                    Command::Audio {
                        mode: AudioCommand::Silent
                    }
                ) == silent
            );
        }
    }
}
