#![cfg_attr(not(test), no_std)]
#![cfg_attr(not(test), no_main)]

mod arch;
mod body;
mod build_identity;
mod capabilities;
mod commands;
#[cfg(any(feature = "pico-w", test))]
mod dhcp;
mod drivers;
mod events;
mod hardware;
#[cfg(any(feature = "pico-w", test))]
mod icmp;
mod network_registry;
mod runtime;
mod session;
mod status;

#[cfg(all(not(test), not(feature = "pico-w")))]
use panic_halt as _;

#[cfg(all(not(test), feature = "pico-w"))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo<'_>) -> ! {
    // RP2040 SIO GPIO_OUT_CLR. Fail the external reset gate inactive even if
    // firmware panics during the asserted portion of a RUN pulse.
    unsafe { core::ptr::write_volatile(0xd000_0018 as *mut u32, 1 << 21) };
    cortex_m::interrupt::disable();
    loop {
        cortex_m::asm::wfi();
    }
}

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
