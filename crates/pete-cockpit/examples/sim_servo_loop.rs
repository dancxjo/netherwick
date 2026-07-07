use pete_cockpit::{Cockpit, EventCursor, SimCockpit};

fn main() -> pete_cockpit::Result<()> {
    let mut cockpit = SimCockpit::new();
    let mut cursor = EventCursor::new();

    let caps = cockpit.get_capabilities()?;
    println!(
        "sim capabilities body={} drive={} verbs={}",
        caps.body_kind,
        caps.drive,
        caps.verbs.join(",")
    );

    cockpit.arm()?;
    println!("accepted arm");
    print_events("after arm", &mut cursor, &mut cockpit)?;

    cockpit.heartbeat_stop(250)?;
    println!("accepted heartbeat_stop ttl_ms=250");
    cockpit.cmd_vel(70, 0, 200)?;
    println!("accepted cmd_vel linear_mm_s=70 angular_mrad_s=0 ttl_ms=200");
    print_events("after cmd_vel", &mut cursor, &mut cockpit)?;

    cockpit.advance_ms(200);
    print_events("after ttl completion", &mut cursor, &mut cockpit)?;

    cockpit.cmd_vel(40, 0, 500)?;
    println!("accepted cmd_vel linear_mm_s=40 angular_mrad_s=0 ttl_ms=500");
    cockpit.trip_safety();
    println!("simulated safety stop");
    let batch = print_events("after safety stop", &mut cursor, &mut cockpit)?;
    if batch.has_stop_reason() {
        println!("stop reason observed");
    }

    let status = cockpit.get_status()?;
    println!("final status {}", status.raw);
    Ok(())
}

fn print_events(
    label: &str,
    cursor: &mut EventCursor,
    cockpit: &mut SimCockpit,
) -> pete_cockpit::Result<pete_cockpit::EventBatch> {
    let before = cursor.next_seq();
    let batch = cursor.poll(cockpit)?;
    println!(
        "{label}: cursor {} -> {} ({} events)",
        before,
        cursor.next_seq(),
        batch.events.len()
    );
    for event in &batch.events {
        println!("event {} {:?}", event.seq, event.kind);
    }
    Ok(batch)
}
