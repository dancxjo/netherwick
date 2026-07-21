#[allow(clippy::too_many_arguments)]
fn spawn_wifi_lane(
    pio0: Peri<'static, PIO0>,
    dma0: Peri<'static, DMA_CH0>,
    wifi_power: Peri<'static, PIN_23>,
    wifi_dio: Peri<'static, PIN_24>,
    wifi_cs: Peri<'static, PIN_25>,
    wifi_clk: Peri<'static, PIN_29>,
    forebrain_uart: Peri<'static, UART1>,
    forebrain_tx: Peri<'static, PIN_4>,
    forebrain_rx: Peri<'static, PIN_5>,
    i2c1: Peri<'static, I2C1>,
    i2c_sda: Peri<'static, PIN_2>,
    i2c_scl: Peri<'static, PIN_3>,
    usb: Peri<'static, USB>,
) -> ! {
    static EXECUTOR: StaticCell<embassy_executor::Executor> = StaticCell::new();
    let executor = EXECUTOR.init(embassy_executor::Executor::new());
    executor.run(|spawner| {
        spawner.spawn(
            wifi_task(spawner, pio0, dma0, wifi_power, wifi_dio, wifi_cs, wifi_clk)
                .expect("spawn wifi task"),
        );
        spawner.spawn(
            forebrain_uart_task(forebrain_uart, forebrain_tx, forebrain_rx)
                .expect("spawn forebrain uart task"),
        );
        spawner.spawn(i2c_sensor_task(i2c1, i2c_sda, i2c_scl).expect("spawn I2C task"));
        spawn_usb_cdc_tasks(&spawner, usb);
    })
}

#[embassy_executor::task]
#[allow(clippy::too_many_arguments)]
async fn wifi_task(
    spawner: Spawner,
    pio0: Peri<'static, PIO0>,
    dma0: Peri<'static, DMA_CH0>,
    wifi_power: Peri<'static, PIN_23>,
    wifi_dio: Peri<'static, PIN_24>,
    wifi_cs: Peri<'static, PIN_25>,
    wifi_clk: Peri<'static, PIN_29>,
) {
    status::mark_wifi_starting();
    if let Some((stack, mut control)) =
        start_wifi_ap(spawner, pio0, dma0, wifi_power, wifi_dio, wifi_cs, wifi_clk).await
    {
        status::mark_wifi_ap_started();
        let _ = control.gpio_set(0, false).await;
        for _ in 0..HTTP_TASKS {
            spawner.spawn(http_task(stack).expect("spawn http task"));
        }
        spawner.spawn(websocket_task(stack).expect("spawn websocket task"));
        spawner.spawn(udp_control_task(stack).expect("spawn udp control task"));
        spawner.spawn(dns_task(stack).expect("spawn dns task"));
        spawner.spawn(mdns_task(stack).expect("spawn mdns task"));
        spawner.spawn(dhcp_task(stack).expect("spawn dhcp task"));
        status::mark_wifi_services_started();
        onboard_led_loop(&mut control).await;
    }

    status::mark_wifi_error();
    loop {
        Timer::after_secs(LED_HEARTBEAT_INTERVAL_SECS).await;
    }
}

async fn start_wifi_ap(
    spawner: Spawner,
    pio0: Peri<'static, PIO0>,
    dma0: Peri<'static, DMA_CH0>,
    wifi_power: Peri<'static, PIN_23>,
    wifi_dio: Peri<'static, PIN_24>,
    wifi_cs: Peri<'static, PIN_25>,
    wifi_clk: Peri<'static, PIN_29>,
) -> Option<(Stack<'static>, cyw43::Control<'static>)> {
    let fw = aligned_bytes!("../../firmware/cyw43/43439A0.bin");
    let clm = aligned_bytes!("../../firmware/cyw43/43439A0_clm.bin");
    let nvram = aligned_bytes!("../../firmware/cyw43/nvram_rp2040.bin");

    let pwr = Output::new(wifi_power, Level::Low);
    let cs = Output::new(wifi_cs, Level::High);
    let mut pio = Pio::new(pio0, Irqs);
    let spi = PioSpi::new(
        &mut pio.common,
        pio.sm0,
        DEFAULT_CLOCK_DIVIDER,
        pio.irq0,
        cs,
        wifi_dio,
        wifi_clk,
        dma::Channel::new(dma0, Irqs),
    );

    static STATE: StaticCell<cyw43::State> = StaticCell::new();
    let state = STATE.init(cyw43::State::new());
    let (net_device, mut control, runner) = cyw43::new(state, pwr, spi, fw, nvram).await;
    spawner.spawn(cyw43_runner_task(runner).expect("spawn cyw43 runner"));

    control.init(clm).await;
    control
        .set_power_management(cyw43::PowerManagementMode::None)
        .await;

    let config = NetConfig::ipv4_static(embassy_net::StaticConfigV4 {
        address: Ipv4Cidr::new(AP_IP, 24),
        dns_servers: Default::default(),
        gateway: None,
    });

    static RESOURCES: StaticCell<StackResources<10>> = StaticCell::new();
    let (stack, runner) = embassy_net::new(
        IcmpEchoDevice::new(net_device),
        config,
        RESOURCES.init(StackResources::new()),
        0x5eed,
    );
    let _ = stack.join_multicast_group(IpAddress::Ipv4(Ipv4Address::new(224, 0, 0, 251)));
    spawner.spawn(net_runner_task(runner).expect("spawn net runner"));

    let hardware_address = stack.hardware_address();
    AP_INSTANCE_ID.store(stable_instance_id(hardware_address), Ordering::Release);
    let ssid = ap_ssid(hardware_address);
    control.start_ap_open(&ssid, AP_CHANNEL).await;
    Some((stack, control))
}

fn ap_ssid(address: HardwareAddress) -> heapless::String<9> {
    let mut ssid = heapless::String::<9>::new();
    let _ = ssid.push_str(AP_SSID_PREFIX);
    let mut value = stable_instance_id(address);
    let mut digits = [b'0'; 4];
    for digit in digits.iter_mut().rev() {
        let remainder = (value % INSTANCE_ID_BASE) as u8;
        *digit = if remainder < 10 {
            b'0' + remainder
        } else {
            b'a' + (remainder - 10)
        };
        value /= INSTANCE_ID_BASE;
    }
    for digit in digits {
        let _ = ssid.push(digit as char);
    }
    ssid
}

fn stable_instance_id(address: HardwareAddress) -> u32 {
    let mut hash = 0x811c_9dc5u32;
    let HardwareAddress::Ethernet(address) = address;
    for byte in address.as_bytes() {
        hash ^= *byte as u32;
        hash = hash.wrapping_mul(0x0100_0193);
    }
    hash % INSTANCE_ID_MODULUS
}

fn stable_board_id(unique_id: &[u8]) -> u32 {
    let mut hash = 0x811c_9dc5u32;
    for byte in unique_id {
        hash ^= *byte as u32;
        hash = hash.wrapping_mul(0x0100_0193);
    }
    hash
}

fn boot_entropy() -> u32 {
    let mut value = 0u32;
    for _ in 0..32 {
        for _ in 0..37 {
            cortex_m::asm::nop();
        }
        value = (value << 1) | rp_pac::ROSC.randombit().read().randombit() as u32;
    }
    value ^ Instant::now().as_micros() as u32
}

#[embassy_executor::task]
async fn cyw43_runner_task(
    runner: cyw43::Runner<'static, cyw43::SpiBus<Output<'static>, PioSpi<'static, PIO0, 0>>>,
) -> ! {
    runner.run().await
}

#[embassy_executor::task]
async fn net_runner_task(mut runner: embassy_net::Runner<'static, IcmpEchoDevice<'static>>) -> ! {
    runner.run().await
}
