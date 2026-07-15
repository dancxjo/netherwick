use pete_cockpit::{Cockpit, CockpitResponse, HttpCockpit, UartCockpit};

fn main() -> pete_cockpit::Result<()> {
    let _ = dotenvy::dotenv();
    let args = std::env::args().collect::<Vec<_>>();
    let transport = args.get(1).map(String::as_str).unwrap_or("uart");
    let endpoint = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("/dev/serial/by-id/usb-Pete_Robotics_Pete_Brainstem_Cockpit_c8aa51e4-if00");

    match transport {
        "uart" => debug(UartCockpit::connect(endpoint)?),
        "http" => debug(HttpCockpit::connect(endpoint)),
        other => {
            eprintln!("usage: brainstem_debug [uart PATH|http HOST:PORT]");
            Err(pete_cockpit::CockpitError::BadResponse(format!(
                "unknown transport {other}"
            )))
        }
    }
}

fn debug<C: Cockpit>(mut cockpit: C) -> pete_cockpit::Result<()> {
    let status = cockpit.get_status()?;
    println!("status.raw={}", status.raw);
    println!("status.summary={:#?}", status.summary());

    let capabilities = cockpit.get_capabilities()?;
    println!(
        "capabilities body={} drive={} limits={:?}",
        capabilities.body_kind, capabilities.drive, capabilities.limits
    );

    let event_next_seq = status.summary().event_next_seq.unwrap_or(0);
    let since_seq = event_next_seq.saturating_sub(32);
    let events = cockpit.get_events_since(since_seq)?;
    println!(
        "events since={} next={} dropped_before={} count={}",
        events.since_seq,
        events.next_seq,
        events.dropped_before_seq,
        events.events.len()
    );
    for event in events.events {
        println!(
            "event seq={} kind={} a={} b={} c={}",
            event.seq,
            event.kind.as_str(),
            event.a,
            event.b,
            event.c
        );
    }

    if let CockpitResponse::Status(status) =
        cockpit.execute(pete_cockpit::CockpitRequest::GetStatus)?
    {
        println!("status.after.raw={}", status.raw);
    }
    Ok(())
}
