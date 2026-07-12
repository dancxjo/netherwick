use std::net::ToSocketAddrs;
use std::thread;
use std::time::Duration;

use pete_cockpit::{
    AddressFamily, Cockpit, CockpitRequest, ControlAuthority, MotherbrainBootstrap,
    RegisterNetworkEndpoint,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::dotenv();
    let args = std::env::args().collect::<Vec<_>>();
    let smoke = args.iter().any(|arg| arg == "--possess-smoke");
    let lease_expiry_smoke = args.iter().any(|arg| arg == "--lease-expiry-smoke");
    let identity_only = args.iter().any(|arg| arg == "--identity-only");
    let wheels_off_floor = args.iter().any(|arg| arg == "--wheels-off-floor");
    if smoke && !wheels_off_floor {
        return Err("--possess-smoke requires --wheels-off-floor".into());
    }
    let mut bootstrap = MotherbrainBootstrap::from_host();
    bootstrap.expected_brainstem_device_id = std::env::var("PETE_BRAINSTEM_DEVICE_ID").ok();
    let mut ready = bootstrap.connect_usb()?;
    eprintln!(
        "brainstem identity: {}\nbrainstem boot: {}\nprotocol: {}.{}\nsession: {}",
        ready.session().peer_device_id,
        ready.session().peer_boot_id,
        ready.session().protocol_major,
        ready.session().protocol_minor,
        ready.session().session_id,
    );
    eprintln!("capabilities: valid");
    let welcome_safety = &ready.outcome().welcome.safety_snapshot;
    eprintln!(
        "safety: stopped={}, disarmed={}, estop={}",
        !welcome_safety.active_motion, !welcome_safety.armed, welcome_safety.estop_latched
    );
    if identity_only {
        eprintln!("wired motherbrain identity established");
        return Ok(());
    }

    if let (Ok(address), Ok(lease_identity)) = (
        std::env::var("PETE_BRAINSTEM_WIFI_IPV4"),
        std::env::var("PETE_DHCP_CLIENT_ID_HEX"),
    ) {
        let registered = bootstrap.register_network(
            &mut ready,
            RegisterNetworkEndpoint {
                interface_id: std::env::var("PETE_BRAINSTEM_INTERFACE")
                    .unwrap_or_else(|_| "wlan1".into()),
                address_family: AddressFamily::Ipv4,
                address,
                hostname: "motherbrain".into(),
                lease_identity,
                ttl_seconds: 600,
            },
        )?;
        let resolved = (registered.fqdn.as_str(), 80)
            .to_socket_addrs()
            .ok()
            .and_then(|mut addresses| addresses.next());
        if resolved.is_none() {
            return Err(format!("DNS verification failed for {}", registered.fqdn).into());
        }
        eprintln!("network: {} -> {}", registered.fqdn, registered.address);
    }

    let lease = ready
        .acquire_control(
            ControlAuthority::Motherbrain,
            if lease_expiry_smoke { 300 } else { 5_000 },
        )?
        .clone();
    eprintln!(
        "control lease: {} generation={} ttl_ms={}",
        lease.lease_id, lease.generation, lease.ttl_ms
    );
    if lease_expiry_smoke {
        let session = ready.session().clone();
        thread::sleep(Duration::from_millis(500));
        let expired = ready.connector_mut().execute_with_lease(
            &session,
            &lease,
            CockpitRequest::HeartbeatStop { timeout_ms: 500 },
        );
        if expired.is_ok() {
            return Err("expired control lease was unexpectedly accepted".into());
        }
        eprintln!("expired lease: rejected");

        let fresh = ready
            .acquire_control(ControlAuthority::Motherbrain, 2_000)?
            .clone();
        if fresh.generation <= lease.generation || fresh.lease_id == lease.lease_id {
            return Err("fresh control lease did not advance identity and generation".into());
        }
        eprintln!(
            "fresh control lease: {} generation={} ttl_ms={}",
            fresh.lease_id, fresh.generation, fresh.ttl_ms
        );
        let stale = ready.connector_mut().execute_with_lease(
            &session,
            &lease,
            CockpitRequest::HeartbeatStop { timeout_ms: 500 },
        );
        if stale.is_ok() {
            return Err("superseded control lease was unexpectedly accepted".into());
        }
        eprintln!("superseded lease: rejected");
        ready.execute(CockpitRequest::HeartbeatStop { timeout_ms: 500 })?;
        eprintln!("fresh lease-bound heartbeat stop: accepted");
        thread::sleep(Duration::from_millis(600));
        ready.execute(CockpitRequest::Stop)?;
        let status = ready.execute(CockpitRequest::GetStatus)?;
        let pete_cockpit::CockpitResponse::Status(status) = status else {
            return Err("final status response was malformed".into());
        };
        let summary = status.summary();
        if summary.active_motion == Some(true) {
            return Err("lease expiry smoke did not finish stopped".into());
        }
        eprintln!("lease expiry smoke complete: stopped; body remains brainstem-supervised");
    } else if smoke {
        ready.execute(CockpitRequest::CmdVel {
            linear_mm_s: 50,
            angular_mrad_s: 0,
            ttl_ms: 125,
        })?;
        thread::sleep(Duration::from_millis(250));
        ready.execute(CockpitRequest::Stop)?;
        let events = ready.poll_events()?;
        for event in &events.events {
            eprintln!("event {}: {}", event.seq, event.kind.as_str());
        }
        let status = ready.execute(CockpitRequest::GetStatus)?;
        let pete_cockpit::CockpitResponse::Status(status) = status else {
            return Err("final status response was malformed".into());
        };
        let summary = status.summary();
        if summary.active_motion != Some(false) {
            return Err("possession smoke did not finish stopped".into());
        }
        eprintln!("possession smoke complete: stopped, disarmed");
    } else {
        eprintln!("bootstrap complete: ready, disarmed");
    }
    Ok(())
}
