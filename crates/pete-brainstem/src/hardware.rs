use crate::drivers::imu::{ImuHealth, ImuSample};

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum SerialRead {
    Byte(u8),
    WouldBlock,
    Error(UartReadError),
}

#[derive(Clone, Copy, Eq, PartialEq)]
#[allow(dead_code)]
pub enum UartReadError {
    Overrun,
    Break,
    Parity,
    Framing,
    Other,
}

pub trait BrainstemHardware {
    fn delay_ms(&mut self, ms: u32);
    fn now_us(&mut self) -> u32;
    fn feed_watchdog(&mut self);

    fn set_power_toggle(&mut self, high: bool);
    fn set_brc(&mut self, high: bool);
    fn set_indicators(&mut self, on: bool);
    #[allow(dead_code)]
    fn set_primary_indicator(&mut self, on: bool);

    /// Drives the external open-drain stage connected across the Pi 5 RUN
    /// header. `true` asserts reset; the Pico must never drive RUN high.
    fn set_motherbrain_reset(&mut self, _asserted: bool) {}

    fn write_byte(&mut self, byte: u8) -> Result<(), ()>;
    fn flush_uart(&mut self) -> Result<(), ()>;
    fn read_byte(&mut self) -> SerialRead;

    fn set_create_uart_baud(&mut self, _baud: u32) -> Result<(), ()> {
        Err(())
    }

    fn poll_imu_sample(&mut self, _now_ms: u32) -> Result<Option<ImuSample>, ImuHealth> {
        Ok(None)
    }

    fn charging_indicator_active(&mut self) -> Option<bool> {
        None
    }

    fn restart_imu(&mut self) -> Result<(), ImuHealth> {
        Err(ImuHealth::Absent)
    }

    fn drain_uart_rx(&mut self) {
        for _ in 0..256 {
            match self.read_byte() {
                SerialRead::Byte(_) | SerialRead::Error(_) => {}
                SerialRead::WouldBlock => break,
            }
        }
    }
}
