#![cfg_attr(not(any(test, feature = "rpi5")), no_std)]
#![cfg_attr(not(any(test, feature = "rpi5")), no_main)]

#[cfg(all(not(test), not(feature = "pico-w"), not(feature = "rpi5")))]
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
    pete_brainstem::arch::rp2040::run()
}

#[cfg(all(not(test), feature = "pico-w"))]
#[cortex_m_rt::entry]
fn main() -> ! {
    let peripherals = embassy_rp::init(Default::default());
    pete_brainstem::arch::pico_w::spawn_safety_lane(peripherals)
}

#[cfg(all(not(test), feature = "rpi5"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    pete_brainstem::run_rpi5(pete_brainstem::Rpi5Config::from_env()?)
}
