use core::convert::Infallible;

use embedded_hal::delay::DelayNs;
use embedded_hal::digital::OutputPin;
use embedded_hal_nb::serial::{Read, Write};
use fugit::RateExtU32;
use nb::block;
use rp2040_hal as hal;

use hal::clocks::{init_clocks_and_plls, Clock};
use hal::gpio::bank0::{Gpio18, Gpio19, Gpio20, Gpio25};
use hal::gpio::{FunctionSioOutput, FunctionUart, Pin, PullDown};
use hal::pac;
use hal::sio::Sio;
use hal::uart::{DataBits, Enabled, StopBits, UartConfig, UartPeripheral};
use hal::watchdog::Watchdog;

use crate::body;
use crate::hardware::{BrainstemHardware, SerialRead};

#[link_section = ".boot2"]
#[used]
static BOOT2: [u8; 256] = rp2040_boot2::BOOT_LOADER_W25Q080;

const CREATE_UART_DATA_BITS: DataBits = DataBits::Eight;
const CREATE_UART_STOP_BITS: StopBits = StopBits::One;

// Unsafe hardware assumptions for this RP2040/Pico backend:
// - body.toml maps GP16/Pico physical pin 21 to Create RX.
// - body.toml maps GP17/Pico physical pin 22 from Create TX through external 5V-to-3.3V level shifting.
// - body.toml maps GP18 to the external Create Power Toggle interface with the correct polarity and isolation.
// - body.toml maps GP19 to Create BRC; BRC is released high by this firmware.
// - body.toml maps GP20 as an optional external LED output; leave unconnected if unused.

type CreateUart = UartPeripheral<
    Enabled,
    pac::UART0,
    (
        Pin<hal::gpio::bank0::Gpio16, FunctionUart, PullDown>,
        Pin<hal::gpio::bank0::Gpio17, FunctionUart, PullDown>,
    ),
>;

type Output<P> = Pin<P, FunctionSioOutput, PullDown>;

pub struct Rp2040Brainstem {
    timer: hal::Timer,
    uart: CreateUart,
    power_toggle: Output<Gpio18>,
    brc: Output<Gpio19>,
    external_led: Output<Gpio20>,
    onboard_led: Output<Gpio25>,
}

impl Rp2040Brainstem {
    pub fn new() -> Self {
        let mut pac = pac::Peripherals::take().unwrap();
        let mut watchdog = Watchdog::new(pac.WATCHDOG);
        let clocks = init_clocks_and_plls(
            body::XOSC_CRYSTAL_FREQ_HZ,
            pac.XOSC,
            pac.CLOCKS,
            pac.PLL_SYS,
            pac.PLL_USB,
            &mut pac.RESETS,
            &mut watchdog,
        )
        .ok()
        .unwrap();

        let timer = hal::Timer::new(pac.TIMER, &mut pac.RESETS, &clocks);
        let sio = Sio::new(pac.SIO);
        let pins = hal::gpio::Pins::new(
            pac.IO_BANK0,
            pac.PADS_BANK0,
            sio.gpio_bank0,
            &mut pac.RESETS,
        );

        let uart_pins = (
            pins.gpio16.into_function::<FunctionUart>(),
            pins.gpio17.into_function::<FunctionUart>(),
        );
        let uart = UartPeripheral::new(pac.UART0, uart_pins, &mut pac.RESETS)
            .enable(
                UartConfig::new(
                    body::CREATE_UART_BAUD.Hz(),
                    CREATE_UART_DATA_BITS,
                    None,
                    CREATE_UART_STOP_BITS,
                ),
                clocks.peripheral_clock.freq(),
            )
            .unwrap();

        Self {
            timer,
            uart,
            power_toggle: pins.gpio18.into_push_pull_output(),
            brc: pins.gpio19.into_push_pull_output(),
            external_led: pins.gpio20.into_push_pull_output(),
            onboard_led: pins.gpio25.into_push_pull_output(),
        }
    }
}

impl BrainstemHardware for Rp2040Brainstem {
    fn delay_ms(&mut self, ms: u32) {
        self.timer.delay_ms(ms);
    }

    fn now_us(&mut self) -> u32 {
        self.timer.get_counter_low()
    }

    fn set_power_toggle(&mut self, high: bool) {
        set_pin(&mut self.power_toggle, high);
    }

    fn set_brc(&mut self, high: bool) {
        set_pin(&mut self.brc, high);
    }

    fn set_indicators(&mut self, on: bool) {
        set_pin(&mut self.onboard_led, on);
        set_pin(&mut self.external_led, on);
    }

    fn set_primary_indicator(&mut self, on: bool) {
        set_pin(&mut self.onboard_led, on);
    }

    fn write_byte(&mut self, byte: u8) -> Result<(), ()> {
        block!(self.uart.write(byte)).map_err(|_| ())
    }

    fn flush_uart(&mut self) -> Result<(), ()> {
        block!(self.uart.flush()).map_err(|_| ())
    }

    fn read_byte(&mut self) -> SerialRead {
        match self.uart.read() {
            Ok(byte) => SerialRead::Byte(byte),
            Err(nb::Error::WouldBlock) => SerialRead::WouldBlock,
            Err(nb::Error::Other(_)) => SerialRead::Error,
        }
    }
}

fn set_pin<P>(pin: &mut P, high: bool)
where
    P: OutputPin<Error = Infallible>,
{
    if high {
        let _ = pin.set_high();
    } else {
        let _ = pin.set_low();
    }
}
