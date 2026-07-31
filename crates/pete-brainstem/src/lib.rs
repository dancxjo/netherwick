#![cfg_attr(not(any(test, feature = "rpi5")), no_std)]

pub mod arch;
mod audio;
mod body;
mod build_identity;
mod capabilities;
mod commands;
pub mod conduit_network;
#[cfg(any(feature = "pico-w", test))]
mod dhcp;
#[cfg(any(feature = "pico-w", test))]
mod display;
mod drivers;
mod events;
mod hardware;
#[cfg(any(feature = "pico-w", test))]
mod icmp;
mod network_registry;
#[cfg(feature = "rpi5")]
mod rpi5_control;
mod runtime;
mod session;
mod status;

#[cfg(feature = "rpi5")]
pub use arch::rpi5::{run as run_rpi5, Rpi5Config};
