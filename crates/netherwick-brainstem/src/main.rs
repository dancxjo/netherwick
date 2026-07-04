#![no_std]
#![no_main]

use core::convert::Infallible;

use embedded_hal::delay::DelayNs;
use embedded_hal::digital::OutputPin;
use fugit::RateExtU32;
use nb::block;
use panic_halt as _;
use rp2040_hal as hal;

use hal::clocks::{init_clocks_and_plls, Clock};
use hal::gpio::FunctionUart;
use hal::pac;
use hal::sio::Sio;
use hal::uart::{DataBits, StopBits, UartConfig, UartPeripheral};
use hal::watchdog::Watchdog;

#[link_section = ".boot2"]
#[used]
static BOOT2: [u8; 256] = rp2040_boot2::BOOT_LOADER_W25Q080;

const XOSC_CRYSTAL_FREQ_HZ: u32 = 12_000_000;

const CREATE_UART_BAUD: u32 = 57_600;
const CREATE_UART_DATA_BITS: DataBits = DataBits::Eight;
const CREATE_UART_STOP_BITS: StopBits = StopBits::One;

#[allow(dead_code)]
mod pinout {
    pub const CREATE_TX_GPIO: u8 = 16;
    pub const CREATE_TX_PHYSICAL_PIN: u8 = 21;
    pub const CREATE_RX_GPIO: u8 = 17;
    pub const CREATE_RX_PHYSICAL_PIN: u8 = 22;
    pub const ONBOARD_LED_GPIO: u8 = 25;
    pub const CREATE_POWER_TOGGLE_GPIO: u8 = 18;
    pub const CREATE_BRC_GPIO: u8 = 19;
    pub const EXTERNAL_LED_GPIO: u8 = 20;
}

const BOOT_LED_BLINKS: usize = 3;
const POWER_TOGGLE_PULSE_MS: u32 = 500;
const CREATE_WAKE_WAIT_MS: u32 = 3_000;
const CREATE_RESPONSIVE_TIMEOUT_MS: u32 = 6_000;
const BRC_LOW_PULSE_MS: u32 = 500;
const POST_BRC_SETTLE_MS: u32 = 100;
const POST_MODE_SETTLE_MS: u32 = 100;
const DRIVE_STEP_MS: u32 = 350;
const FINAL_POWER_CYCLE_WAIT_MS: u32 = 1_000;
const IDLE_BLINK_MS: u32 = 1_000;
const ERROR_BLINK_MS: u32 = 150;
const ERROR_PAUSE_MS: u32 = 900;
const UART_BYTE_TIMEOUT_MS: u32 = 20;

const OI_START: u8 = 128;
const OI_SAFE: u8 = 131;
const OI_SENSORS: u8 = 142;
const OI_PACKET_BATTERY_CHARGE: u8 = 25;
const OI_DRIVE: u8 = 137;

const DEMO_MODE: CreateMode = CreateMode::Safe;
const JIG_VELOCITY_MM_S: i16 = 120;
const TURN_VELOCITY_MM_S: i16 = 80;
const DRIVE_STRAIGHT_RADIUS: i16 = 0x8000u16 as i16;
const DRIVE_TURN_LEFT_RADIUS: i16 = 1;
const DRIVE_TURN_RIGHT_RADIUS: i16 = -1;

// Unsafe hardware assumptions:
// - GP16/Pico physical pin 21 is wired to Create RX.
// - GP17/Pico physical pin 22 is wired from Create TX through external 5V-to-3.3V level shifting.
// - GP18 drives the external Create Power Toggle interface with the correct polarity and isolation.
// - GP19 drives Create BRC; BRC is released high by this firmware.
// - GP20 is optional external LED output; leave unconnected if unused.

#[derive(Clone, Copy)]
enum CreateMode {
    Safe,
    #[allow(dead_code)]
    Full,
}

impl CreateMode {
    const fn command(self) -> u8 {
        match self {
            Self::Safe => OI_SAFE,
            Self::Full => 132,
        }
    }
}

#[hal::entry]
fn main() -> ! {
    let mut pac = pac::Peripherals::take().unwrap();
    let mut watchdog = Watchdog::new(pac.WATCHDOG);
    let clocks = init_clocks_and_plls(
        XOSC_CRYSTAL_FREQ_HZ,
        pac.XOSC,
        pac.CLOCKS,
        pac.PLL_SYS,
        pac.PLL_USB,
        &mut pac.RESETS,
        &mut watchdog,
    )
    .ok()
    .unwrap();

    let mut timer = hal::Timer::new(pac.TIMER, &mut pac.RESETS, &clocks);
    let sio = Sio::new(pac.SIO);
    let pins = hal::gpio::Pins::new(
        pac.IO_BANK0,
        pac.PADS_BANK0,
        sio.gpio_bank0,
        &mut pac.RESETS,
    );

    let mut onboard_led = pins.gpio25.into_push_pull_output();
    let mut external_led = pins.gpio20.into_push_pull_output();
    let mut power_toggle = pins.gpio18.into_push_pull_output();
    let mut brc = pins.gpio19.into_push_pull_output();

    let uart_pins = (
        pins.gpio16.into_function::<FunctionUart>(),
        pins.gpio17.into_function::<FunctionUart>(),
    );
    let mut create_uart = UartPeripheral::new(pac.UART0, uart_pins, &mut pac.RESETS)
        .enable(
            UartConfig::new(
                CREATE_UART_BAUD.Hz(),
                CREATE_UART_DATA_BITS,
                None,
                CREATE_UART_STOP_BITS,
            ),
            clocks.peripheral_clock.freq(),
        )
        .unwrap();

    set_low(&mut power_toggle);
    set_high(&mut brc);
    boot_indicator(&mut timer, &mut onboard_led, &mut external_led);

    if let Err(()) = run_create_demo(&mut timer, &mut create_uart, &mut power_toggle, &mut brc) {
        fail_safe(
            &mut timer,
            &mut create_uart,
            &mut onboard_led,
            &mut external_led,
        );
    }

    set_low(&mut onboard_led);
    set_low(&mut external_led);
    loop {
        set_high(&mut onboard_led);
        timer.delay_ms(IDLE_BLINK_MS);
        set_low(&mut onboard_led);
        timer.delay_ms(IDLE_BLINK_MS);
    }
}

fn run_create_demo<UART, POWER, BRC>(
    timer: &mut hal::Timer,
    uart: &mut UART,
    power_toggle: &mut POWER,
    brc: &mut BRC,
) -> Result<(), ()>
where
    UART: embedded_hal_nb::serial::Read<u8> + embedded_hal_nb::serial::Write<u8>,
    POWER: OutputPin<Error = Infallible>,
    BRC: OutputPin<Error = Infallible>,
{
    pulse_high(timer, power_toggle, POWER_TOGGLE_PULSE_MS);
    timer.delay_ms(CREATE_WAKE_WAIT_MS);

    wait_for_uart_response(timer, uart, CREATE_RESPONSIVE_TIMEOUT_MS)?;

    set_low(brc);
    timer.delay_ms(BRC_LOW_PULSE_MS);
    set_high(brc);
    timer.delay_ms(POST_BRC_SETTLE_MS);

    send_byte(uart, OI_START)?;
    timer.delay_ms(POST_MODE_SETTLE_MS);
    send_byte(uart, DEMO_MODE.command())?;
    timer.delay_ms(POST_MODE_SETTLE_MS);

    drive(uart, JIG_VELOCITY_MM_S, DRIVE_STRAIGHT_RADIUS)?;
    timer.delay_ms(DRIVE_STEP_MS);
    drive(uart, TURN_VELOCITY_MM_S, DRIVE_TURN_LEFT_RADIUS)?;
    timer.delay_ms(DRIVE_STEP_MS);
    drive(uart, TURN_VELOCITY_MM_S, DRIVE_TURN_RIGHT_RADIUS)?;
    timer.delay_ms(DRIVE_STEP_MS);
    stop_create(uart)?;

    timer.delay_ms(FINAL_POWER_CYCLE_WAIT_MS);
    pulse_high(timer, power_toggle, POWER_TOGGLE_PULSE_MS);
    stop_create(uart)?;
    Ok(())
}

fn wait_for_uart_response<UART>(
    timer: &mut hal::Timer,
    uart: &mut UART,
    timeout_ms: u32,
) -> Result<(), ()>
where
    UART: embedded_hal_nb::serial::Read<u8> + embedded_hal_nb::serial::Write<u8>,
{
    let deadline = timer.get_counter_low().wrapping_add(timeout_ms * 1_000);
    loop {
        send_byte(uart, OI_SENSORS)?;
        send_byte(uart, OI_PACKET_BATTERY_CHARGE)?;
        if read_exact_with_timeout(timer, uart, &mut [0; 2], UART_BYTE_TIMEOUT_MS).is_ok() {
            return Ok(());
        }
        if timer.get_counter_low().wrapping_sub(deadline) < u32::MAX / 2 {
            return Err(());
        }
        timer.delay_ms(100);
    }
}

fn read_exact_with_timeout<UART>(
    timer: &mut hal::Timer,
    uart: &mut UART,
    buf: &mut [u8],
    byte_timeout_ms: u32,
) -> Result<(), ()>
where
    UART: embedded_hal_nb::serial::Read<u8>,
{
    for slot in buf.iter_mut() {
        let start = timer.get_counter_low();
        loop {
            match uart.read() {
                Ok(byte) => {
                    *slot = byte;
                    break;
                }
                Err(nb::Error::WouldBlock) => {
                    if timer.get_counter_low().wrapping_sub(start) >= byte_timeout_ms * 1_000 {
                        return Err(());
                    }
                }
                Err(nb::Error::Other(_)) => return Err(()),
            }
        }
    }
    Ok(())
}

fn drive<UART>(uart: &mut UART, velocity_mm_s: i16, radius_mm: i16) -> Result<(), ()>
where
    UART: embedded_hal_nb::serial::Write<u8>,
{
    let velocity = velocity_mm_s.to_be_bytes();
    let radius = radius_mm.to_be_bytes();
    send_bytes(
        uart,
        &[OI_DRIVE, velocity[0], velocity[1], radius[0], radius[1]],
    )
}

fn stop_create<UART>(uart: &mut UART) -> Result<(), ()>
where
    UART: embedded_hal_nb::serial::Write<u8>,
{
    drive(uart, 0, 0)
}

fn send_byte<UART>(uart: &mut UART, byte: u8) -> Result<(), ()>
where
    UART: embedded_hal_nb::serial::Write<u8>,
{
    block!(uart.write(byte)).map_err(|_| ())?;
    block!(uart.flush()).map_err(|_| ())
}

fn send_bytes<UART>(uart: &mut UART, bytes: &[u8]) -> Result<(), ()>
where
    UART: embedded_hal_nb::serial::Write<u8>,
{
    for byte in bytes {
        block!(uart.write(*byte)).map_err(|_| ())?;
    }
    block!(uart.flush()).map_err(|_| ())
}

fn pulse_high<PIN>(timer: &mut hal::Timer, pin: &mut PIN, pulse_ms: u32)
where
    PIN: OutputPin<Error = Infallible>,
{
    set_high(pin);
    timer.delay_ms(pulse_ms);
    set_low(pin);
}

fn boot_indicator<LedA, LedB>(timer: &mut hal::Timer, led_a: &mut LedA, led_b: &mut LedB)
where
    LedA: OutputPin<Error = Infallible>,
    LedB: OutputPin<Error = Infallible>,
{
    for _ in 0..BOOT_LED_BLINKS {
        set_high(led_a);
        set_high(led_b);
        timer.delay_ms(120);
        set_low(led_a);
        set_low(led_b);
        timer.delay_ms(120);
    }
    set_high(led_a);
    set_high(led_b);
    timer.delay_ms(250);
}

fn fail_safe<UART, LedA, LedB>(
    timer: &mut hal::Timer,
    uart: &mut UART,
    led_a: &mut LedA,
    led_b: &mut LedB,
) -> !
where
    UART: embedded_hal_nb::serial::Write<u8>,
    LedA: OutputPin<Error = Infallible>,
    LedB: OutputPin<Error = Infallible>,
{
    let _ = stop_create(uart);
    loop {
        for _ in 0..3 {
            set_high(led_a);
            set_high(led_b);
            timer.delay_ms(ERROR_BLINK_MS);
            set_low(led_a);
            set_low(led_b);
            timer.delay_ms(ERROR_BLINK_MS);
        }
        timer.delay_ms(ERROR_PAUSE_MS);
    }
}

fn set_high<PIN>(pin: &mut PIN)
where
    PIN: OutputPin<Error = Infallible>,
{
    let _ = pin.set_high();
}

fn set_low<PIN>(pin: &mut PIN)
where
    PIN: OutputPin<Error = Infallible>,
{
    let _ = pin.set_low();
}
