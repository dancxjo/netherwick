use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream, ToSocketAddrs, UdpSocket};
use std::path::{Path, PathBuf};
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
