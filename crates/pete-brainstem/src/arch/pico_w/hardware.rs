struct IcmpEchoDevice<'a> {
    inner: cyw43::NetDriver<'a>,
    inbound: [u8; NETWORK_FRAME_CAPACITY],
    rate_limit: IcmpRateLimit,
}

impl<'a> IcmpEchoDevice<'a> {
    fn new(inner: cyw43::NetDriver<'a>) -> Self {
        Self {
            inner,
            inbound: [0; NETWORK_FRAME_CAPACITY],
            rate_limit: IcmpRateLimit::new(),
        }
    }
}

struct StagedRxToken<'a> {
    frame: &'a mut [u8],
}

impl embassy_net_driver::RxToken for StagedRxToken<'_> {
    fn consume<R, F>(self, f: F) -> R
    where
        F: FnOnce(&mut [u8]) -> R,
    {
        f(self.frame)
    }
}

impl<'d> NetDriver for IcmpEchoDevice<'d> {
    type RxToken<'a>
        = StagedRxToken<'a>
    where
        Self: 'a;
    type TxToken<'a>
        = <cyw43::NetDriver<'d> as NetDriver>::TxToken<'a>
    where
        Self: 'a;

    fn receive(
        &mut self,
        cx: &mut core::task::Context,
    ) -> Option<(Self::RxToken<'_>, Self::TxToken<'_>)> {
        let (inner, inbound, rate_limit) =
            (&mut self.inner, &mut self.inbound, &mut self.rate_limit);
        let (rx, tx) = inner.receive(cx)?;
        let len = rx.consume(|frame| {
            if frame.len() > inbound.len() {
                None
            } else {
                inbound[..frame.len()].copy_from_slice(frame);
                Some(frame.len())
            }
        });
        let len = match len {
            Some(len) => len,
            None => {
                status::mark_icmp_dropped();
                return None;
            }
        };

        match process_icmp_echo_frame(
            &mut inbound[..len],
            Instant::now().as_millis() as u64,
            rate_limit,
        ) {
            IcmpEchoDisposition::NotIcmp => Some((
                StagedRxToken {
                    frame: &mut inbound[..len],
                },
                tx,
            )),
            IcmpEchoDisposition::Reply(reply_len) => {
                status::mark_icmp_echo_request();
                tx.consume(reply_len, |reply| {
                    reply.copy_from_slice(&inbound[..reply_len])
                });
                status::mark_icmp_echo_reply();
                None
            }
            IcmpEchoDisposition::RateLimited => {
                status::mark_icmp_echo_request();
                status::mark_icmp_rate_limited();
                None
            }
            IcmpEchoDisposition::Dropped => {
                status::mark_icmp_dropped();
                None
            }
        }
    }

    fn transmit(&mut self, cx: &mut core::task::Context) -> Option<Self::TxToken<'_>> {
        self.inner.transmit(cx)
    }

    fn link_state(&mut self, cx: &mut core::task::Context) -> embassy_net_driver::LinkState {
        self.inner.link_state(cx)
    }

    fn capabilities(&self) -> embassy_net_driver::Capabilities {
        self.inner.capabilities()
    }

    fn hardware_address(&self) -> embassy_net_driver::HardwareAddress {
        self.inner.hardware_address()
    }
}

type UsbDriver = embassy_rp::usb::Driver<'static, USB>;
type BrainstemUsbDevice = UsbDevice<'static, UsbDriver>;

bind_interrupts!(struct Irqs {
    PIO0_IRQ_0 => PioInterruptHandler<PIO0>;
    DMA_IRQ_0 => dma::InterruptHandler<DMA_CH0>;
    I2C1_IRQ => I2cInterruptHandler<I2C1>;
    USBCTRL_IRQ => embassy_rp::usb::InterruptHandler<USB>;
});

pub struct PicoWBrainstem {
    uart: Uart<'static, Blocking>,
    power_toggle: Output<'static>,
    _txs_oe: Output<'static>,
    status_led: Output<'static>,
    charging_indicator: Input<'static>,
    #[cfg(motherbrain_reset_hardware)]
    motherbrain_reset: Output<'static>,
}

const _: () = assert!(body::CREATE_CHARGING_INDICATOR_GPIO == 20);
const _: () = assert!(body::CREATE_CHARGING_INDICATOR_ACTIVE_HIGH);
const _: () = assert!(body::EXTERNAL_LED_GPIO == 17);

impl PicoWBrainstem {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        uart0: Peri<'static, UART0>,
        tx: Peri<'static, PIN_0>,
        rx: Peri<'static, PIN_1>,
        power_toggle: Peri<'static, PIN_18>,
        txs_oe: Peri<'static, PIN_19>,
        status_led: Peri<'static, PIN_17>,
        charging_indicator: Peri<'static, PIN_20>,
        #[cfg(motherbrain_reset_hardware)] motherbrain_reset: Peri<
            'static,
            embassy_rp::peripherals::PIN_21,
        >,
    ) -> Self {
        let (power_toggle, txs_oe) = initialize_power_control(
            power_toggle,
            txs_oe,
            |pin| Ok::<_, Infallible>(Output::new(pin, Level::Low)),
            |pin| Ok::<_, Infallible>(Output::new(pin, Level::High)),
        )
        .unwrap();
        let mut uart_config = UartConfig::default();
        uart_config.baudrate = body::CREATE_UART_BAUD;
        uart_config.data_bits = DataBits::DataBits8;
        uart_config.stop_bits = StopBits::STOP1;
        uart_config.parity = Parity::ParityNone;
        let uart = Uart::new_blocking(uart0, tx, rx, uart_config);
        // UART idles high. Keep the Create RX input at a defined idle level
        // while the robot or level shifter is unpowered instead of accepting
        // the RP2040 pad's reset pull-down as break/framing noise.
        rp_pac::PADS_BANK0.gpio(1).modify(|w| {
            w.set_pue(true);
            w.set_pde(false);
        });
        Self {
            uart,
            power_toggle,
            _txs_oe: txs_oe,
            status_led: Output::new(status_led, Level::Low),
            charging_indicator: Input::new(
                charging_indicator,
                if body::CREATE_CHARGING_INDICATOR_ACTIVE_HIGH {
                    Pull::Down
                } else {
                    Pull::Up
                },
            ),
            // The gate/base has an external pull-down. Construct it inactive
            // before any command transport starts.
            #[cfg(motherbrain_reset_hardware)]
            motherbrain_reset: Output::new(motherbrain_reset, Level::Low),
        }
    }
}

impl BrainstemHardware for PicoWBrainstem {
    fn delay_ms(&mut self, ms: u32) {
        embassy_time::block_for(Duration::from_millis(ms as u64));
    }

    fn now_us(&mut self) -> u32 {
        Instant::now().as_micros() as u32
    }

    fn feed_watchdog(&mut self) {
        // Watchdog plumbing is owned by the runtime safety lane; this Pico W
        // backend currently leaves the hardware watchdog disabled.
    }

    fn begin_power_toggle_pulse(&mut self) {
        self.power_toggle.set_low();
        self.power_toggle.set_high();
    }

    fn end_power_toggle_pulse(&mut self) {
        self.power_toggle.set_low();
    }

    fn set_indicators(&mut self, on: bool) {
        self.status_led.set_level(level(on));
    }

    fn set_primary_indicator(&mut self, on: bool) {
        self.status_led.set_level(level(on));
    }

    fn set_motherbrain_reset(&mut self, asserted: bool) {
        #[cfg(motherbrain_reset_hardware)]
        self.motherbrain_reset.set_level(level(asserted));
        #[cfg(not(motherbrain_reset_hardware))]
        let _ = asserted;
    }

    fn write_byte(&mut self, byte: u8) -> Result<(), ()> {
        self.uart.blocking_write(&[byte]).map_err(|_| ())
    }

    fn flush_uart(&mut self) -> Result<(), ()> {
        self.uart.blocking_flush().map_err(|_| ())
    }

    fn read_byte(&mut self) -> SerialRead {
        match self.uart.read() {
            Ok(byte) => SerialRead::Byte(byte),
            Err(nb::Error::WouldBlock) => SerialRead::WouldBlock,
            Err(nb::Error::Other(error)) => SerialRead::Error(map_uart_error(error)),
        }
    }

    fn set_create_uart_baud(&mut self, baud: u32) -> Result<(), ()> {
        self.uart.set_baudrate(baud);
        Ok(())
    }

    fn charging_indicator_active(&mut self) -> Option<bool> {
        if body::CREATE_CHARGING_INDICATOR_ENABLED {
            Some(self.charging_indicator.is_high() == body::CREATE_CHARGING_INDICATOR_ACTIVE_HIGH)
        } else {
            None
        }
    }
}

fn map_uart_error(error: UartError) -> UartReadError {
    match error {
        UartError::Overrun => UartReadError::Overrun,
        UartError::Break => UartReadError::Break,
        UartError::Parity => UartReadError::Parity,
        UartError::Framing => UartReadError::Framing,
        _ => UartReadError::Other,
    }
}

pub fn spawn_safety_lane(p: embassy_rp::Peripherals) -> ! {
    let mut flash = embassy_rp::flash::Flash::<_, _, { 2 * 1024 * 1024 }>::new_blocking(p.FLASH);
    let mut unique_id = [0u8; 8];
    let _ = flash.blocking_unique_id(&mut unique_id);
    let instance = stable_board_id(&unique_id).max(1);
    BRAINSTEM_INSTANCE_ID.store(instance, Ordering::Release);
    BRAINSTEM_BOOT_ID.store(
        instance.rotate_left(11) ^ boot_entropy().max(1),
        Ordering::Release,
    );
    let hardware = PicoWBrainstem::new(
        p.UART0,
        p.PIN_0,
        p.PIN_1,
        p.PIN_18,
        p.PIN_19,
        p.PIN_17,
        p.PIN_20,
        #[cfg(motherbrain_reset_hardware)]
        p.PIN_21,
    );

    spawn_core1(
        p.CORE1,
        unsafe { &mut *core::ptr::addr_of_mut!(CORE1_STACK) },
        move || Runtime::new(hardware).run(),
    );

    spawn_wifi_lane(
        p.PIO0, p.DMA_CH0, p.PIN_23, p.PIN_24, p.PIN_25, p.PIN_29, p.UART1, p.PIN_4, p.PIN_5,
        p.I2C1, p.PIN_2, p.PIN_3, p.USB,
    );
}
