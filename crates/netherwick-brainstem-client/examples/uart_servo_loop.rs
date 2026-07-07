use std::thread;
use std::time::{Duration, Instant};

use netherwick_brainstem_client::{
    BrainstemClient, EventCursor, UartBrainstemClient, UartBrainstemClientConfig,
};

fn main() -> netherwick_brainstem_client::Result<()> {
    let path = std::env::args()
        .nth(1)
        .or_else(|| std::env::var("NETHERWICK_BRAINSTEM_UART").ok())
        .unwrap_or_else(|| "/dev/ttyACM0".to_owned());
    let baud_rate = std::env::args()
        .nth(2)
        .or_else(|| std::env::var("NETHERWICK_BRAINSTEM_BAUD").ok())
        .map(|value| value.parse().expect("baud rate must be a u32"))
        .unwrap_or(115_200);

    let config = UartBrainstemClientConfig::new(path).with_baud_rate(baud_rate);
    let mut brainstem = UartBrainstemClient::connect_with_config(config)?;
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

    let run_result = run_servo_loop(&mut brainstem, &mut events);

    let _ = brainstem.stop();
    let _ = brainstem.stream_sensors(false, 0, 0);
    let _ = brainstem.disarm();

    run_result
}

fn run_servo_loop(
    brainstem: &mut UartBrainstemClient,
    events: &mut EventCursor,
) -> netherwick_brainstem_client::Result<()> {
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(3) {
        brainstem.heartbeat_stop(900)?;
        brainstem.cmd_vel(70, 0, 300)?;

        let batch = events.poll(brainstem)?;
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

    Ok(())
}
