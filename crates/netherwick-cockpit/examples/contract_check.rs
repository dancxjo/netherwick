use std::env;
use std::net::SocketAddr;

use netherwick_cockpit::{
    Cockpit, CockpitContract, HttpCockpit, Result, SimCockpit, UartCockpit, UdpCockpit,
    WebSocketCockpit,
};

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let transport = args.get(1).map(String::as_str).unwrap_or("sim");
    match transport {
        "sim" => check("sim", SimCockpit::new()),
        "udp" => {
            let addr: SocketAddr = args
                .get(2)
                .expect("usage: contract_check udp HOST:PORT")
                .parse()
                .expect("UDP address must be HOST:PORT");
            check("udp", UdpCockpit::connect(addr)?)
        }
        "uart" => {
            let path = args
                .get(2)
                .expect("usage: contract_check uart /dev/ttyACM0");
            check("uart", UartCockpit::connect(path)?)
        }
        "http" => {
            let host = args.get(2).expect("usage: contract_check http HOST:PORT");
            check("http", HttpCockpit::connect(host))
        }
        "ws" | "websocket" => {
            let url = args
                .get(2)
                .expect("usage: contract_check ws ws://HOST:81/control");
            check("websocket", WebSocketCockpit::connect_url(url)?)
        }
        other => {
            eprintln!("unknown transport {other}");
            eprintln!("usage: contract_check [sim|udp HOST:PORT|uart PATH|http HOST:PORT|ws URL]");
            Ok(())
        }
    }
}

fn check<C: Cockpit>(label: &str, mut cockpit: C) -> Result<()> {
    let capabilities = cockpit.get_capabilities()?;
    let contract = CockpitContract::new(capabilities);
    let report = contract.validate_local_model();
    println!(
        "{label}: body={} drive={}",
        contract.capabilities().body_kind,
        contract.capabilities().drive
    );
    println!("verbs={}", contract.capabilities().verbs.join(","));
    println!("events={}", contract.capabilities().events.join(","));
    println!(
        "limits linear={} angular={} ttl={}..={}",
        contract.capabilities().limits.max_linear_mm_s,
        contract.capabilities().limits.max_angular_mrad_s,
        contract.capabilities().limits.min_ttl_ms,
        contract.capabilities().limits.max_ttl_ms
    );
    println!("missing verbs: {}", display_list(&report.missing_verbs));
    println!("extra verbs: {}", display_list(&report.extra_verbs));
    println!(
        "optional absent verbs: {}",
        display_list(&report.optional_absent_verbs)
    );
    println!("unknown events: {}", display_list(&report.unknown_events));
    Ok(())
}

fn display_list(values: &[String]) -> String {
    if values.is_empty() {
        "none".to_owned()
    } else {
        values.join(",")
    }
}
