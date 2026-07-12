use pete_cockpit::{
    establish_session, CockpitRequest, HandshakeHello, HttpCockpit, MotherbrainBootstrap,
    ServiceScope,
};

fn main() -> pete_cockpit::Result<()> {
    if std::env::var("PETE_BOOTSEL_USB").as_deref() == Ok("1") {
        eprintln!("maintenance handshake: USB CDC");
        let mut ready = MotherbrainBootstrap::from_host()
            .connect_usb()
            .map_err(|error| pete_cockpit::CockpitError::BadResponse(error.to_string()))?;
        eprintln!(
            "brainstem identity: {} boot={}",
            ready.session().peer_device_id,
            ready.session().peer_boot_id
        );
        let lease = ready.acquire_service(ServiceScope::Bootsel, 5_000)?.clone();
        eprintln!(
            "BOOTSEL service lease: {} generation={} ttl_ms={}",
            lease.lease_id, lease.generation, lease.ttl_ms
        );
        ready.execute(CockpitRequest::Bootsel)?;
        eprintln!("bootsel_accepted; waiting for RPI-RP2");
        return Ok(());
    }
    let host =
        std::env::var("PETE_BRAINSTEM_HTTP_HOST").unwrap_or_else(|_| "192.168.4.1:80".into());
    eprintln!("maintenance handshake: http://{host}");
    let connector = HttpCockpit::connect(host);
    let mut ready = establish_session(connector, HandshakeHello::default_motherbrain(), None)?;
    eprintln!(
        "brainstem identity: {} boot={}",
        ready.session().peer_device_id,
        ready.session().peer_boot_id
    );
    let lease = ready.acquire_service(ServiceScope::Bootsel, 5_000)?.clone();
    eprintln!(
        "BOOTSEL service lease: {} generation={} ttl_ms={}",
        lease.lease_id, lease.generation, lease.ttl_ms
    );
    ready.execute(CockpitRequest::Bootsel)?;
    eprintln!("bootsel_accepted; waiting for RPI-RP2");
    Ok(())
}
