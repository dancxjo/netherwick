use pete_cockpit::{
    establish_session, Cockpit, CockpitRequest, ControlAuthority, HandshakeHello, SessionCockpit,
    SimCockpit,
};

fn main() -> pete_cockpit::Result<()> {
    let cockpit = SimCockpit::new();
    let mut cockpit = establish_session(
        cockpit,
        HandshakeHello::motherbrain("pete-motherbrain-sim-example"),
        None,
    )?;
    let caps = cockpit.contract().capabilities();
    println!(
        "sim capabilities body={} drive={} verbs={}",
        caps.body_kind,
        caps.drive,
        caps.verbs.join(",")
    );

    cockpit.acquire_control(ControlAuthority::Motherbrain, 1_000)?;
    cockpit.control()?.arm()?;
    println!("accepted arm");
    print_events("after arm", &mut cockpit)?;

    cockpit.execute(CockpitRequest::HeartbeatStop { timeout_ms: 250 })?;
    println!("accepted heartbeat_stop ttl_ms=250");
    cockpit.control()?.cmd_vel(70, 0, 200)?;
    println!("accepted cmd_vel linear_mm_s=70 angular_mrad_s=0 ttl_ms=200");
    print_events("after cmd_vel", &mut cockpit)?;

    cockpit.connector_mut().advance_ms(200);
    print_events("after ttl completion", &mut cockpit)?;

    cockpit.control()?.cmd_vel(40, 0, 500)?;
    println!("accepted cmd_vel linear_mm_s=40 angular_mrad_s=0 ttl_ms=500");
    cockpit.connector_mut().trip_safety();
    println!("simulated safety stop");
    let batch = print_events("after safety stop", &mut cockpit)?;
    if batch.has_stop_reason() {
        println!("stop reason observed");
    }

    let status = cockpit.connector_mut().get_status()?;
    println!("final status {}", status.raw);
    Ok(())
}

fn print_events(
    label: &str,
    cockpit: &mut SessionCockpit<SimCockpit>,
) -> pete_cockpit::Result<pete_cockpit::EventBatch> {
    let batch = cockpit.poll_events()?;
    println!(
        "{label}: next cursor {} ({} events)",
        batch.next_seq,
        batch.events.len()
    );
    for event in &batch.events {
        println!("event {} {:?}", event.seq, event.kind);
    }
    Ok(batch)
}
