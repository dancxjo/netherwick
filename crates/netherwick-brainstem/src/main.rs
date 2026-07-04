#![no_std]
#![no_main]

mod arch;
mod body;
mod commands;
mod drivers;
mod events;
mod hardware;
mod runtime;

use panic_halt as _;

#[rp2040_hal::entry]
fn main() -> ! {
    runtime::Runtime::new(arch::rp2040::Rp2040Brainstem::new()).run_demo();
}
