#[embassy_executor::task]
async fn i2c_sensor_task(
    i2c1: Peri<'static, I2C1>,
    sda: Peri<'static, PIN_2>,
    scl: Peri<'static, PIN_3>,
) -> ! {
    let mut config = I2cConfig::default();
    config.frequency = SENSOR_I2C_FREQUENCY_HZ;
    let mut i2c = I2c::new_async(i2c1, scl, sda, Irqs, config);
    let mut imu_address = None;
    let mut next_imu_poll_ms = 0u32;
    let mut next_imu_retry_ms = 0u32;
    let mut oled = OledService::new();

    if !body::IMU_ENABLED {
        status::mark_imu_health(ImuHealth::Absent);
    }

    loop {
        let now_ms = Instant::now().as_millis() as u32;

        if body::IMU_ENABLED {
            if let Some(active_address) = imu_address {
                if deadline_reached(now_ms, next_imu_poll_ms) {
                    let mut bytes = [0u8; 14];
                    if i2c_write_read_with_timeout(
                        &mut i2c,
                        active_address,
                        MPU6050_ACCEL_XOUT_H,
                        &mut bytes,
                    )
                    .await
                    {
                        status::mark_imu_sample(decode_mpu6050_sample(now_ms, &bytes));
                    } else {
                        status::mark_imu_health(ImuHealth::Fault);
                        imu_address = None;
                        next_imu_retry_ms = now_ms.wrapping_add(IMU_RETRY_MS as u32);
                    }
                    next_imu_poll_ms = now_ms.wrapping_add(body::IMU_POLL_PERIOD_MS.max(1));
                }
            } else if deadline_reached(now_ms, next_imu_retry_ms) {
                match initialize_imu(&mut i2c).await {
                    Ok(found_address) => {
                        imu_address = Some(found_address);
                        next_imu_poll_ms = now_ms;
                    }
                    Err(health) => {
                        status::mark_imu_health(health);
                        next_imu_retry_ms = now_ms.wrapping_add(IMU_RETRY_MS as u32);
                    }
                }
            }
        }

        oled.poll(&mut i2c, now_ms).await;
        Timer::after_millis(1).await;
    }
}

async fn initialize_imu(i2c: &mut I2c<'static, I2C1, I2cAsync>) -> Result<u8, ImuHealth> {
    for address in [MPU6050_ADDRESS_LOW, MPU6050_ADDRESS_HIGH] {
        let mut who_am_i = [0u8; 1];
        if !i2c_write_read_with_timeout(i2c, address, MPU6050_WHO_AM_I, &mut who_am_i).await
            || (who_am_i[0] != 0x68 && who_am_i[0] != 0x70)
        {
            continue;
        }

        for command in [
            [MPU6050_PWR_MGMT_1, 0x00],
            [MPU6050_GYRO_CONFIG, 0x00],
            [MPU6050_ACCEL_CONFIG, 0x00],
        ] {
            if !i2c_write_with_timeout(i2c, address, &command).await {
                return Err(ImuHealth::Fault);
            }
        }
        return Ok(address);
    }
    Err(ImuHealth::Absent)
}

async fn i2c_write_with_timeout(
    i2c: &mut I2c<'static, I2C1, I2cAsync>,
    address: u8,
    bytes: &[u8],
) -> bool {
    match select(
        i2c.write_async(address, bytes.iter().copied()),
        Timer::after_millis(SENSOR_I2C_TIMEOUT_MS),
    )
    .await
    {
        Either::First(Ok(())) => true,
        Either::First(Err(_)) | Either::Second(()) => false,
    }
}

async fn i2c_write_read_with_timeout(
    i2c: &mut I2c<'static, I2C1, I2cAsync>,
    address: u8,
    register: u8,
    bytes: &mut [u8],
) -> bool {
    match select(
        i2c.write_read_async(address, [register], bytes),
        Timer::after_millis(SENSOR_I2C_TIMEOUT_MS),
    )
    .await
    {
        Either::First(Ok(())) => true,
        Either::First(Err(_)) | Either::Second(()) => false,
    }
}

struct OledService {
    address: Option<u8>,
    next_probe_index: usize,
    next_action_ms: u32,
    next_refresh_ms: u32,
    transfer_offset: Option<usize>,
    desired_frame: [u8; display::FRAMEBUFFER_BYTES],
    sent_frame: [u8; display::FRAMEBUFFER_BYTES],
}

impl OledService {
    const fn new() -> Self {
        Self {
            address: None,
            next_probe_index: 0,
            next_action_ms: 0,
            next_refresh_ms: 0,
            transfer_offset: None,
            desired_frame: [0; display::FRAMEBUFFER_BYTES],
            sent_frame: [0xff; display::FRAMEBUFFER_BYTES],
        }
    }

    async fn poll(&mut self, i2c: &mut I2c<'static, I2C1, I2cAsync>, now_ms: u32) {
        if !deadline_reached(now_ms, self.next_action_ms) {
            return;
        }

        let Some(address) = self.address else {
            let candidate = OLED_ADDRESSES[self.next_probe_index];
            if oled_write_with_timeout(i2c, candidate, &OLED_INIT_COMMANDS).await {
                self.address = Some(candidate);
                self.next_probe_index = 0;
                self.next_refresh_ms = now_ms;
                self.next_action_ms = now_ms;
            } else if self.next_probe_index + 1 < OLED_ADDRESSES.len() {
                self.next_probe_index += 1;
                self.next_action_ms = now_ms.wrapping_add(OLED_IO_INTERVAL_MS);
            } else {
                self.next_probe_index = 0;
                self.next_action_ms = now_ms.wrapping_add(OLED_RETRY_MS);
            }
            return;
        };

        if self.transfer_offset.is_none() && deadline_reached(now_ms, self.next_refresh_ms) {
            let snapshot = status::snapshot(now_ms);
            let ap_instance_id = AP_INSTANCE_ID.load(Ordering::Acquire);
            let page = DisplayStatus::from_snapshot(
                &snapshot,
                DisplayNetwork {
                    ssid_suffix: (ap_instance_id != AP_INSTANCE_UNKNOWN).then_some(ap_instance_id),
                    active_leases: network_registry::diagnostics(now_ms).active_leases,
                },
            )
            .page(DisplaySafety::current(), now_ms);
            self.desired_frame = display::render(&page);
            self.next_refresh_ms = now_ms.wrapping_add(display::REFRESH_PERIOD_MS);
            if self.desired_frame != self.sent_frame {
                self.transfer_offset = Some(0);
            }
        }

        let Some(offset) = self.transfer_offset else {
            self.next_action_ms = self.next_refresh_ms;
            return;
        };
        let end = (offset + OLED_CHUNK_BYTES).min(display::FRAMEBUFFER_BYTES);
        let page = offset / display::WIDTH;
        let column = offset % display::WIDTH;
        let mut bytes = [0u8; OLED_CHUNK_BYTES + 7];
        bytes[0] = 0x80;
        bytes[1] = 0xb0 | page as u8;
        bytes[2] = 0x80;
        bytes[3] = (column & 0x0f) as u8;
        bytes[4] = 0x80;
        bytes[5] = 0x10 | ((column >> 4) & 0x0f) as u8;
        bytes[6] = 0x40;
        bytes[7..7 + end - offset].copy_from_slice(&self.desired_frame[offset..end]);

        if oled_write_with_timeout(i2c, address, &bytes[..7 + end - offset]).await {
            self.sent_frame[offset..end].copy_from_slice(&self.desired_frame[offset..end]);
            self.transfer_offset = (end < display::FRAMEBUFFER_BYTES).then_some(end);
            self.next_action_ms = now_ms.wrapping_add(OLED_IO_INTERVAL_MS);
        } else {
            self.address = None;
            self.next_probe_index = 0;
            self.transfer_offset = None;
            self.sent_frame = [0xff; display::FRAMEBUFFER_BYTES];
            self.next_action_ms = now_ms.wrapping_add(OLED_RETRY_MS);
        }
    }
}

async fn oled_write_with_timeout(
    i2c: &mut I2c<'static, I2C1, I2cAsync>,
    address: u8,
    bytes: &[u8],
) -> bool {
    match select(
        i2c.write_async(address, bytes.iter().copied()),
        Timer::after_millis(OLED_I2C_TIMEOUT_MS),
    )
    .await
    {
        Either::First(Ok(())) => true,
        Either::First(Err(_)) | Either::Second(()) => false,
    }
}

fn deadline_reached(now_ms: u32, deadline_ms: u32) -> bool {
    now_ms.wrapping_sub(deadline_ms) < u32::MAX / 2
}

#[embassy_executor::task]
async fn forebrain_uart_task(
    uart1: Peri<'static, UART1>,
    tx: Peri<'static, PIN_4>,
    rx: Peri<'static, PIN_5>,
) -> ! {
    let mut uart_config = UartConfig::default();
    uart_config.baudrate = FOREBRAIN_UART_BAUD;
    uart_config.data_bits = DataBits::DataBits8;
    uart_config.stop_bits = StopBits::STOP1;
    uart_config.parity = Parity::ParityNone;

    let mut uart = Uart::new_blocking(uart1, tx, rx, uart_config);
    let mut line = heapless::Vec::<u8, FOREBRAIN_LINE_MAX>::new();
    let mut line_started_ms = 0;

    loop {
        let now_ms = Instant::now().as_millis() as u32;
        match uart.read() {
            Ok(byte) => {
                status::mark_forebrain_uart_rx_byte(now_ms);
                if line.is_empty() {
                    line_started_ms = now_ms;
                }

                match byte {
                    b'\r' => {}
                    b'\n' => {
                        handle_forebrain_uart_line(&mut uart, &line);
                        line.clear();
                        line_started_ms = 0;
                    }
                    byte => {
                        if line.push(byte).is_err() {
                            line.clear();
                            line_started_ms = 0;
                            status::mark_forebrain_uart_error(
                                status::ForebrainUartErrorCode::LineTooLong,
                            );
                            submit_forebrain_stop();
                            write_forebrain_uart_line(&mut uart, b"ERR 0 line_too_long\n");
                        }
                    }
                }
            }
            Err(nb::Error::WouldBlock) => {
                if !line.is_empty()
                    && now_ms.wrapping_sub(line_started_ms) > FOREBRAIN_LINE_TIMEOUT_MS
                {
                    line.clear();
                    line_started_ms = 0;
                    status::mark_forebrain_uart_error(status::ForebrainUartErrorCode::Parse);
                    submit_forebrain_stop();
                    write_forebrain_uart_line(&mut uart, b"ERR 0 timeout\n");
                }
                Timer::after_millis(FOREBRAIN_POLL_MS).await;
            }
            Err(nb::Error::Other(_)) => {
                line.clear();
                line_started_ms = 0;
                status::mark_forebrain_uart_error(status::ForebrainUartErrorCode::Uart);
                submit_forebrain_stop();
                write_forebrain_uart_line(&mut uart, b"ERR 0 uart\n");
                Timer::after_millis(FOREBRAIN_POLL_MS).await;
            }
        }
    }
}

fn spawn_usb_cdc_tasks(spawner: &Spawner, usb: Peri<'static, USB>) {
    let driver = embassy_rp::usb::Driver::new(usb, Irqs);
    let mut config = embassy_usb::Config::new(0x1209, 0x5054);
    config.manufacturer = Some("Pete Robotics");
    config.product = Some("Pete Brainstem Cockpit");
    let mut serial = heapless::String::<24>::new();
    let _ = write!(
        serial,
        "{:08x}",
        BRAINSTEM_INSTANCE_ID.load(Ordering::Acquire)
    );
    static SERIAL_NUMBER: StaticCell<heapless::String<24>> = StaticCell::new();
    config.serial_number = Some(SERIAL_NUMBER.init(serial).as_str());
    config.max_power = 100;
    config.max_packet_size_0 = 64;

    static CONFIG_DESCRIPTOR: StaticCell<[u8; 256]> = StaticCell::new();
    static BOS_DESCRIPTOR: StaticCell<[u8; 256]> = StaticCell::new();
    static CONTROL_BUF: StaticCell<[u8; 64]> = StaticCell::new();
    static CDC_STATE: StaticCell<CdcAcmState<'static>> = StaticCell::new();
    let mut builder = embassy_usb::Builder::new(
        driver,
        config,
        CONFIG_DESCRIPTOR.init([0; 256]),
        BOS_DESCRIPTOR.init([0; 256]),
        &mut [],
        CONTROL_BUF.init([0; 64]),
    );
    let class = CdcAcmClass::new(&mut builder, CDC_STATE.init(CdcAcmState::new()), 64);
    let device = builder.build();
    spawner.spawn(usb_device_task(device).expect("spawn USB device task"));
    spawner.spawn(usb_cdc_task(class).expect("spawn USB CDC task"));
}

#[embassy_executor::task]
async fn usb_device_task(mut device: BrainstemUsbDevice) -> ! {
    device.run().await
}

#[embassy_executor::task]
async fn usb_cdc_task(mut class: CdcAcmClass<'static, UsbDriver>) -> ! {
    let mut packet = [0u8; 64];
    let mut line = heapless::Vec::<u8, FOREBRAIN_LINE_MAX>::new();
    let mut response = heapless::String::<4096>::new();
    loop {
        class.wait_connection().await;
        line.clear();
        loop {
            let len = match class.read_packet(&mut packet).await {
                Ok(len) => len,
                Err(_) => break,
            };
            for byte in &packet[..len] {
                match *byte {
                    b'\r' => {}
                    b'\n' => {
                        if let Ok(command) = core::str::from_utf8(&line) {
                            response.clear();
                            let boot_to_usb = handle_compact_control_line(
                                command,
                                &mut response,
                                TransportKind::UsbCdc as u8,
                            )
                            .unwrap_or(false);
                            for chunk in response.as_bytes().chunks(64) {
                                if class.write_packet(chunk).await.is_err() {
                                    break;
                                }
                            }
                            if boot_to_usb {
                                Timer::after_millis(100).await;
                                reset_to_usb_boot(0, 0);
                            }
                        }
                        line.clear();
                    }
                    byte => {
                        if line.push(byte).is_err() {
                            line.clear();
                            let _ = class.write_packet(b"ERR 0 line_too_long\n").await;
                        }
                    }
                }
            }
        }
    }
}
