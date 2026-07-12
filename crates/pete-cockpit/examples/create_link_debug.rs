use std::thread;
use std::time::Duration;

use pete_cockpit::{
    Cockpit, CockpitRequest, ControlAuthority, FeedbackKind, MotherbrainBootstrap,
    PowerStateRequest,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::dotenv();
    let wake = std::env::args().any(|arg| arg == "--wake");
    let mut ready = MotherbrainBootstrap::from_host().connect_usb()?;
    ready.acquire_control(ControlAuthority::Motherbrain, 30_000)?;

    print_status(&mut ready, "before")?;
    if wake {
        ready.execute(CockpitRequest::PowerState {
            request: PowerStateRequest::Wake,
        })?;
        eprintln!("explicit wake requested; waiting for power pulse and Create startup");
        thread::sleep(Duration::from_secs(12));
        print_status(&mut ready, "after-explicit-wake")?;
    }

    // This probe never drives the wheels. The brainstem asserts Start + Full,
    // then validates packet 35 inside a header/length/checksum-framed stream.
    for (label, request) in [
        ("baud-19200", PowerStateRequest::DebugBaud19200),
        ("baud-57600", PowerStateRequest::DebugBaud57600),
        ("baud-115200", PowerStateRequest::DebugBaud115200),
    ] {
        ready.execute(CockpitRequest::PowerState { request })?;
        thread::sleep(Duration::from_millis(100));
        ready.execute(CockpitRequest::PowerState {
            request: PowerStateRequest::StartOi,
        })?;
        thread::sleep(Duration::from_millis(1_500));
        print_status(&mut ready, label)?;
    }

    ready.execute(CockpitRequest::PowerState {
        request: PowerStateRequest::DebugBaud57600,
    })?;
    thread::sleep(Duration::from_millis(1_500));
    ready.execute(CockpitRequest::PlayFeedback {
        feedback: FeedbackKind::Armed,
    })?;
    thread::sleep(Duration::from_millis(300));
    print_status(&mut ready, "restored-57600-after-mode-and-song")?;

    let events = ready.poll_events()?;
    for event in events.events {
        eprintln!(
            "event {} {} a={} b={} c={}",
            event.seq,
            event.kind.as_str(),
            event.a,
            event.b,
            event.c
        );
    }
    Ok(())
}

fn print_status<C: Cockpit>(
    ready: &mut pete_cockpit::SessionCockpit<C>,
    label: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let response = ready.execute(CockpitRequest::GetStatus)?;
    let pete_cockpit::CockpitResponse::Status(status) = response else {
        return Err("status response was malformed".into());
    };
    eprintln!("{label}: {}", status.raw);
    Ok(())
}
