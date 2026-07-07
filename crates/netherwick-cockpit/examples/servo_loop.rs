use std::net::SocketAddr;
use std::thread;
use std::time::{Duration, Instant};

use netherwick_cockpit::{Cockpit, EventCursor, UdpCockpit};

fn main() -> netherwick_cockpit::Result<()> {
    let addr: SocketAddr = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "192.168.4.1:82".to_owned())
        .parse()
        .expect("cockpit address must be host:port");

    let mut cockpit = UdpCockpit::connect(addr)?;
    let mut events = EventCursor::new();

    let caps = cockpit.get_capabilities()?;
    println!(
        "cockpit body={} drive={} verbs={}",
        caps.body_kind,
        caps.drive,
        caps.verbs.join(",")
    );

    cockpit.arm()?;
    cockpit.stream_sensors(true, 0, 250)?;

    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(3) {
        cockpit.heartbeat_stop(900)?;
        cockpit.cmd_vel(70, 0, 300)?;

        let batch = events.poll(&mut cockpit)?;
        for event in &batch.events {
            println!("event {} {:?}", event.seq, event.kind);
        }
        if batch.has_stop_reason() {
            eprintln!("stop reason event observed; stopping");
            cockpit.stop()?;
            return Ok(());
        }

        thread::sleep(Duration::from_millis(200));
    }

    cockpit.stop()?;
    Ok(())
}
