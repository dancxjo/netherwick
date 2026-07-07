use std::net::SocketAddr;
use std::thread;
use std::time::{Duration, Instant};

use netherwick_brainstem_client::{BrainstemClient, EventCursor, UdpBrainstemClient};

fn main() -> netherwick_brainstem_client::Result<()> {
    let addr: SocketAddr = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "192.168.4.1:82".to_owned())
        .parse()
        .expect("brainstem address must be host:port");

    let mut brainstem = UdpBrainstemClient::connect(addr)?;
    let mut events = EventCursor::new();

    let caps = brainstem.get_capabilities()?;
    println!(
        "brainstem body={} drive={} verbs={}",
        caps.body_kind,
        caps.drive,
        caps.verbs.join(",")
    );

    brainstem.arm()?;
    brainstem.stream_sensors(true, 0, 250)?;

    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(3) {
        brainstem.heartbeat_stop(900)?;
        brainstem.cmd_vel(70, 0, 300)?;

        let batch = events.poll(&mut brainstem)?;
        for event in &batch.events {
            println!("event {} {:?}", event.seq, event.kind);
        }
        if batch.has_stop_reason() {
            eprintln!("stop reason event observed; stopping");
            brainstem.stop()?;
            return Ok(());
        }

        thread::sleep(Duration::from_millis(200));
    }

    brainstem.stop()?;
    Ok(())
}
