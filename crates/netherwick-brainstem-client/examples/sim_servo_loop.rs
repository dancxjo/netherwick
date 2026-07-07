use netherwick_brainstem_client::{BrainstemClient, EventCursor, SimBrainstemClient};

fn main() -> netherwick_brainstem_client::Result<()> {
    let mut brainstem = SimBrainstemClient::new();
    let mut cursor = EventCursor::new();

    let caps = brainstem.get_capabilities()?;
    println!(
        "sim capabilities body={} drive={} verbs={}",
        caps.body_kind,
        caps.drive,
        caps.verbs.join(",")
    );

    brainstem.arm()?;
    println!("accepted arm");
    print_events("after arm", &mut cursor, &mut brainstem)?;

    brainstem.heartbeat_stop(250)?;
    println!("accepted heartbeat_stop ttl_ms=250");
    brainstem.cmd_vel(70, 0, 200)?;
    println!("accepted cmd_vel linear_mm_s=70 angular_mrad_s=0 ttl_ms=200");
    print_events("after cmd_vel", &mut cursor, &mut brainstem)?;

    brainstem.advance_ms(200);
    print_events("after ttl completion", &mut cursor, &mut brainstem)?;

    brainstem.cmd_vel(40, 0, 500)?;
    println!("accepted cmd_vel linear_mm_s=40 angular_mrad_s=0 ttl_ms=500");
    brainstem.trip_safety();
    println!("simulated safety stop");
    let batch = print_events("after safety stop", &mut cursor, &mut brainstem)?;
    if batch.has_stop_reason() {
        println!("stop reason observed");
    }

    let status = brainstem.get_status()?;
    println!("final status {}", status.raw);
    Ok(())
}

fn print_events(
    label: &str,
    cursor: &mut EventCursor,
    brainstem: &mut SimBrainstemClient,
) -> netherwick_brainstem_client::Result<netherwick_brainstem_client::EventBatch> {
    let before = cursor.next_seq();
    let batch = cursor.poll(brainstem)?;
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
