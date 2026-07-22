use std::collections::HashMap;
use std::future::Future;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream, ToSocketAddrs, UdpSocket};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serialport::SerialPort;
use thiserror::Error;
use tungstenite::stream::MaybeTlsStream;
use tungstenite::{connect, Message, WebSocket};

mod handshake;
pub use handshake::*;
pub use pete_cockpit_protocol::{
    CommandRejectReason, CONTACT_WITHDRAWAL_DURATION_MS, CONTACT_WITHDRAWAL_SPEED_MM_S,
};

pub type Result<T> = std::result::Result<T, CockpitError>;

static PHYSICAL_ACTUATOR_TRANSPORT_DENIALS: AtomicUsize = AtomicUsize::new(0);

struct PhysicalActuatorTransportDenial;

impl PhysicalActuatorTransportDenial {
    fn enter() -> Self {
        PHYSICAL_ACTUATOR_TRANSPORT_DENIALS.fetch_add(1, Ordering::SeqCst);
        Self
    }
}

impl Drop for PhysicalActuatorTransportDenial {
    fn drop(&mut self) {
        PHYSICAL_ACTUATOR_TRANSPORT_DENIALS.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Execute a future under a process-wide fail-closed prohibition on physical
/// actuator transports. Every built-in UART and network cockpit checks this
/// policy before opening, binding, resolving, or connecting a transport.
pub async fn with_physical_actuator_transports_denied<F: Future>(future: F) -> F::Output {
    let _denial = PhysicalActuatorTransportDenial::enter();
    future.await
}

/// Whether the current process is inside a fail-closed physical transport
/// scope. Intended for audit manifests and invariant assertions.
pub fn physical_actuator_transports_are_denied() -> bool {
    PHYSICAL_ACTUATOR_TRANSPORT_DENIALS.load(Ordering::SeqCst) != 0
}

fn ensure_physical_actuator_transport_allowed(kind: &str) -> Result<()> {
    if !physical_actuator_transports_are_denied() {
        Ok(())
    } else {
        Err(CockpitError::Policy(format!(
            "physical actuator transport {kind} is prohibited in this execution scope"
        )))
    }
}

// The public API stays in this namespace; each include owns one conceptual area.
include!("cockpit/interface.rs");
include!("cockpit/contract.rs");
include!("cockpit/wire.rs");
include!("cockpit/safety.rs");
include!("cockpit/events.rs");
include!("cockpit/simulator.rs");
include!("cockpit/possession.rs");

#[cfg(test)]
#[path = "lib_tests.rs"]
mod tests;
