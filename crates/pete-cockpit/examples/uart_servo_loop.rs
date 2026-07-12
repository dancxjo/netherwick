use std::thread;
use std::time::{Duration, Instant};

use pete_cockpit::{Cockpit, EventCursor, UartCockpit, UartCockpitConfig};

fn main() -> pete_cockpit::Result<()> {
    let path = std::env::args()
        .nth(1)
        .or_else(|| std::env::var("PETE_BRAINSTEM_UART").ok())
        .unwrap_or_else(|| "/dev/ttyACM0".to_owned());
    let baud_rate = std::env::args()
        .nth(2)
        .or_else(|| std::env::var("PETE_BRAINSTEM_BAUD").ok())
        .map(|value| value.parse().expect("baud rate must be a u32"))
        .unwrap_or(115_200);

    let config = UartCockpitConfig::new(path).with_baud_rate(baud_rate);
    let mut cockpit = UartCockpit::connect_with_config(config)?;
    let mut events = EventCursor::new();

    let caps = cockpit.get_capabilities()?;
    println!(
        "cockpit body={} drive={} verbs={}",
        caps.body_kind,
        caps.drive,
        caps.verbs.join(",")
    );

    cockpit.stream_sensors(true, 0, 250)?;

    let run_result = run_servo_loop(&mut cockpit, &mut events);

    let _ = cockpit.stop();
    let _ = cockpit.stream_sensors(false, 0, 0);

    run_result
}

fn run_servo_loop(cockpit: &mut UartCockpit, events: &mut EventCursor) -> pete_cockpit::Result<()> {
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(3) {
        cockpit.heartbeat_stop(900)?;
        cockpit.cmd_vel(70, 0, 300)?;

        let batch = events.poll(cockpit)?;
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

    Ok(())
}
