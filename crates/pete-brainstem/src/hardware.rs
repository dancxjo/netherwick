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

    /// Starts one deliberate Create power-toggle pulse.
    ///
    /// Implementations must first drive POWER_TOGGLE low and only then drive
    /// it high. This is intentionally not an arbitrary GPIO setter: every
    /// call represents exactly one requested low-to-high transition.
    fn begin_power_toggle_pulse(&mut self);
    /// Ends a Create power-toggle pulse and leaves POWER_TOGGLE low so the
    /// circuit is armed for the next request.
    fn end_power_toggle_pulse(&mut self);
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

    fn drain_uart_rx(&mut self) {
        for _ in 0..256 {
            match self.read_byte() {
                SerialRead::Byte(_) | SerialRead::Error(_) => {}
                SerialRead::WouldBlock => break,
            }
        }
    }
}

/// Establishes the r23 power-control outputs in their electrically safe order.
///
/// TXS_OE must remain under its external pull-down until POWER_TOGGLE has been
/// configured as a driven-low output. Keeping the two construction operations
/// in this helper makes that dependency explicit for both RP2040 backends and
/// gives host tests a way to prove the order and failure behavior.
pub(crate) fn initialize_power_control<
    PowerTogglePin,
    TxsOePin,
    PowerToggleOutput,
    TxsOeOutput,
    Error,
>(
    power_toggle_pin: PowerTogglePin,
    txs_oe_pin: TxsOePin,
    establish_power_toggle_low: impl FnOnce(PowerTogglePin) -> Result<PowerToggleOutput, Error>,
    enable_txs_oe: impl FnOnce(TxsOePin) -> Result<TxsOeOutput, Error>,
) -> Result<(PowerToggleOutput, TxsOeOutput), Error> {
    let power_toggle = establish_power_toggle_low(power_toggle_pin)?;
    let txs_oe = enable_txs_oe(txs_oe_pin)?;
    Ok((power_toggle, txs_oe))
}

#[cfg(test)]
mod tests {
    use super::initialize_power_control;
    use core::cell::{Cell, RefCell};

    #[test]
    fn power_toggle_is_established_low_before_txs_oe_is_enabled() {
        let step = Cell::new(0);

        let initialized = initialize_power_control(
            (),
            (),
            |_| {
                assert_eq!(step.get(), 0);
                step.set(1);
                Ok::<_, ()>("POWER_TOGGLE low")
            },
            |_| {
                assert_eq!(step.get(), 1);
                step.set(2);
                Ok::<_, ()>("TXS_OE high")
            },
        );

        assert_eq!(initialized, Ok(("POWER_TOGGLE low", "TXS_OE high")));
        assert_eq!(step.get(), 2);
    }

    #[test]
    fn txs_oe_stays_disabled_when_power_toggle_low_initialization_fails() {
        let oe_enable_attempts = Cell::new(0);

        let initialized = initialize_power_control(
            (),
            (),
            |_| Err::<(), _>("POWER_TOGGLE initialization failed"),
            |_| {
                oe_enable_attempts.set(oe_enable_attempts.get() + 1);
                Ok::<_, &str>(())
            },
        );

        assert_eq!(initialized, Err("POWER_TOGGLE initialization failed"));
        assert_eq!(oe_enable_attempts.get(), 0);
    }

    #[test]
    fn txs_oe_is_enabled_once_and_retained_while_power_toggle_pulses() {
        let power_toggle_levels = RefCell::new(std::vec::Vec::new());
        let txs_oe_high = Cell::new(false);
        let txs_oe_high_writes = Cell::new(0);

        let (power_toggle, txs_oe) = initialize_power_control(
            (),
            (),
            |_| {
                power_toggle_levels.borrow_mut().push(false);
                Ok::<_, ()>(&power_toggle_levels)
            },
            |_| {
                txs_oe_high.set(true);
                txs_oe_high_writes.set(txs_oe_high_writes.get() + 1);
                Ok::<_, ()>(&txs_oe_high)
            },
        )
        .unwrap();

        power_toggle.borrow_mut().extend([false, true, false]);

        assert!(txs_oe.get());
        assert_eq!(txs_oe_high_writes.get(), 1);
        assert_eq!(
            power_toggle.borrow().as_slice(),
            &[false, false, true, false]
        );
    }
}
