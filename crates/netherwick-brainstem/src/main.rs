#![no_std]
#![no_main]

mod arch;
mod create;
mod hardware;

use panic_halt as _;

#[rp2040_hal::entry]
fn main() -> ! {
    let mut hardware = arch::rp2040::Rp2040Brainstem::new();

    if create::run_demo(&mut hardware).is_err() {
        create::fail_safe(&mut hardware);
    }

    create::idle(&mut hardware);
}
