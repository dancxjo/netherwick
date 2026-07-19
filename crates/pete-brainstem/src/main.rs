#![cfg_attr(not(any(test, feature = "rpi5")), no_std)]
#![cfg_attr(not(any(test, feature = "rpi5")), no_main)]

#[cfg(all(not(test), not(feature = "rpi5")))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo<'_>) -> ! {
    // RP2040 SIO GPIO_OUT_CLR. Leave POWER_TOGGLE low, disable TXS_OE, and
    // fail the external reset gate inactive before halting.
    unsafe {
        core::ptr::write_volatile(0xd000_0018 as *mut u32, (1 << 18) | (1 << 19) | (1 << 21))
    };
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
