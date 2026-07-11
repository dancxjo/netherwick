#![cfg_attr(not(test), no_std)]
#![cfg_attr(not(test), no_main)]

mod arch;
mod body;
mod capabilities;
mod commands;
#[cfg(any(feature = "pico-w", test))]
mod dhcp;
mod drivers;
mod events;
mod hardware;
mod runtime;
mod status;

#[cfg(not(test))]
use panic_halt as _;

#[cfg(all(not(test), feature = "rp2040", not(feature = "pico-w")))]
#[rp2040_hal::entry]
fn main() -> ! {
    runtime::Runtime::new(arch::rp2040::Rp2040Brainstem::new()).run();
}

#[cfg(all(not(test), feature = "pico-w"))]
#[cortex_m_rt::entry]
fn main() -> ! {
    let peripherals = embassy_rp::init(Default::default());
    arch::pico_w::spawn_safety_lane(peripherals)
}
