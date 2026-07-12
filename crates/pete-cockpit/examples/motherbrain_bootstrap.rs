use std::net::ToSocketAddrs;
use std::thread;
use std::time::Duration;

use pete_cockpit::{
    AddressFamily, CockpitRequest, ControlAuthority, MotherbrainBootstrap, RegisterNetworkEndpoint,
};

fn main() -> pete_cockpit::Result<()> {
    let mut bootstrap = MotherbrainBootstrap::from_host();
    bootstrap.expected_brainstem_device_id = std::env::var("PETE_BRAINSTEM_DEVICE_ID").ok();
    let mut ready = bootstrap.connect_usb()?;
    eprintln!(
        "brainstem device={} boot={} transport=usb",
        ready.session().peer_device_id,
        ready.session().peer_boot_id
    );

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
        eprintln!(
            "brainstem hostname={} registered={} dns_verified={}",
            "brainstem.pete.internal",
            registered.fqdn,
            resolved.is_some()
        );
    }

    let heartbeat_ms = ready.session().heartbeat_ms;
    ready.acquire_control(
        ControlAuthority::Motherbrain,
        heartbeat_ms.saturating_mul(3),
    )?;
    eprintln!("control lease established; robot remains disarmed");
    loop {
        ready.execute(CockpitRequest::HeartbeatStop {
            timeout_ms: heartbeat_ms.saturating_mul(3),
        })?;
        thread::sleep(Duration::from_millis(heartbeat_ms as u64));
    }
}
