#![no_std]
#![no_main]

mod arch;
mod body;
mod capabilities;
mod commands;
mod drivers;
mod events;
mod hardware;
mod runtime;
mod status;

use panic_halt as _;

#[cfg(all(feature = "rp2040", not(feature = "pico-w")))]
#[rp2040_hal::entry]
fn main() -> ! {
    runtime::Runtime::new(arch::rp2040::Rp2040Brainstem::new()).run_demo();
}

#[cfg(feature = "pico-w")]
#[cortex_m_rt::entry]
fn main() -> ! {
    let peripherals = embassy_rp::init(Default::default());
    arch::pico_w::spawn_safety_lane(peripherals)
}
