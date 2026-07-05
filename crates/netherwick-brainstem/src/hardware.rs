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

    fn set_power_toggle(&mut self, high: bool);
    fn set_brc(&mut self, high: bool);
    fn set_indicators(&mut self, on: bool);
    #[allow(dead_code)]
    fn set_primary_indicator(&mut self, on: bool);

    fn write_byte(&mut self, byte: u8) -> Result<(), ()>;
    fn flush_uart(&mut self) -> Result<(), ()>;
    fn read_byte(&mut self) -> SerialRead;
}
