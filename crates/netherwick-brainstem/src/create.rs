use crate::hardware::{BrainstemHardware, SerialRead};

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
const OI_FULL: u8 = 132;
const OI_SENSORS: u8 = 142;
const OI_PACKET_BATTERY_CHARGE: u8 = 25;
const OI_DRIVE: u8 = 137;

const DEMO_MODE: CreateMode = CreateMode::Safe;
const JIG_VELOCITY_MM_S: i16 = 120;
const TURN_VELOCITY_MM_S: i16 = 80;
const DRIVE_STRAIGHT_RADIUS: i16 = 0x8000u16 as i16;
const DRIVE_TURN_LEFT_RADIUS: i16 = 1;
const DRIVE_TURN_RIGHT_RADIUS: i16 = -1;

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
            Self::Full => OI_FULL,
        }
    }
}

pub fn run_demo<H>(hardware: &mut H) -> Result<(), ()>
where
    H: BrainstemHardware,
{
    hardware.set_power_toggle(false);
    hardware.set_brc(true);
    boot_indicator(hardware);

    pulse_power_toggle(hardware);
    hardware.delay_ms(CREATE_WAKE_WAIT_MS);
    wait_for_uart_response(hardware, CREATE_RESPONSIVE_TIMEOUT_MS)?;

    hardware.set_brc(false);
    hardware.delay_ms(BRC_LOW_PULSE_MS);
    hardware.set_brc(true);
    hardware.delay_ms(POST_BRC_SETTLE_MS);

    send_byte(hardware, OI_START)?;
    hardware.delay_ms(POST_MODE_SETTLE_MS);
    send_byte(hardware, DEMO_MODE.command())?;
    hardware.delay_ms(POST_MODE_SETTLE_MS);

    drive(hardware, JIG_VELOCITY_MM_S, DRIVE_STRAIGHT_RADIUS)?;
    hardware.delay_ms(DRIVE_STEP_MS);
    drive(hardware, TURN_VELOCITY_MM_S, DRIVE_TURN_LEFT_RADIUS)?;
    hardware.delay_ms(DRIVE_STEP_MS);
    drive(hardware, TURN_VELOCITY_MM_S, DRIVE_TURN_RIGHT_RADIUS)?;
    hardware.delay_ms(DRIVE_STEP_MS);
    stop_create(hardware)?;

    hardware.delay_ms(FINAL_POWER_CYCLE_WAIT_MS);
    pulse_power_toggle(hardware);
    stop_create(hardware)?;
    Ok(())
}

pub fn idle<H>(hardware: &mut H) -> !
where
    H: BrainstemHardware,
{
    hardware.set_indicators(false);
    loop {
        hardware.set_primary_indicator(true);
        hardware.delay_ms(IDLE_BLINK_MS);
        hardware.set_primary_indicator(false);
        hardware.delay_ms(IDLE_BLINK_MS);
    }
}

pub fn fail_safe<H>(hardware: &mut H) -> !
where
    H: BrainstemHardware,
{
    let _ = stop_create(hardware);
    loop {
        for _ in 0..3 {
            hardware.set_indicators(true);
            hardware.delay_ms(ERROR_BLINK_MS);
            hardware.set_indicators(false);
            hardware.delay_ms(ERROR_BLINK_MS);
        }
        hardware.delay_ms(ERROR_PAUSE_MS);
    }
}

fn wait_for_uart_response<H>(hardware: &mut H, timeout_ms: u32) -> Result<(), ()>
where
    H: BrainstemHardware,
{
    let deadline = hardware.now_us().wrapping_add(timeout_ms * 1_000);
    loop {
        send_byte(hardware, OI_SENSORS)?;
        send_byte(hardware, OI_PACKET_BATTERY_CHARGE)?;
        if read_exact_with_timeout(hardware, &mut [0; 2], UART_BYTE_TIMEOUT_MS).is_ok() {
            return Ok(());
        }
        if hardware.now_us().wrapping_sub(deadline) < u32::MAX / 2 {
            return Err(());
        }
        hardware.delay_ms(100);
    }
}

fn read_exact_with_timeout<H>(
    hardware: &mut H,
    buf: &mut [u8],
    byte_timeout_ms: u32,
) -> Result<(), ()>
where
    H: BrainstemHardware,
{
    for slot in buf.iter_mut() {
        let start = hardware.now_us();
        loop {
            match hardware.read_byte() {
                SerialRead::Byte(byte) => {
                    *slot = byte;
                    break;
                }
                SerialRead::WouldBlock => {
                    if hardware.now_us().wrapping_sub(start) >= byte_timeout_ms * 1_000 {
                        return Err(());
                    }
                }
                SerialRead::Error => return Err(()),
            }
        }
    }
    Ok(())
}

fn drive<H>(hardware: &mut H, velocity_mm_s: i16, radius_mm: i16) -> Result<(), ()>
where
    H: BrainstemHardware,
{
    let velocity = velocity_mm_s.to_be_bytes();
    let radius = radius_mm.to_be_bytes();
    send_bytes(
        hardware,
        &[OI_DRIVE, velocity[0], velocity[1], radius[0], radius[1]],
    )
}

fn stop_create<H>(hardware: &mut H) -> Result<(), ()>
where
    H: BrainstemHardware,
{
    drive(hardware, 0, 0)
}

fn send_byte<H>(hardware: &mut H, byte: u8) -> Result<(), ()>
where
    H: BrainstemHardware,
{
    hardware.write_byte(byte)?;
    hardware.flush_uart()
}

fn send_bytes<H>(hardware: &mut H, bytes: &[u8]) -> Result<(), ()>
where
    H: BrainstemHardware,
{
    for byte in bytes {
        hardware.write_byte(*byte)?;
    }
    hardware.flush_uart()
}

fn pulse_power_toggle<H>(hardware: &mut H)
where
    H: BrainstemHardware,
{
    hardware.set_power_toggle(true);
    hardware.delay_ms(POWER_TOGGLE_PULSE_MS);
    hardware.set_power_toggle(false);
}

fn boot_indicator<H>(hardware: &mut H)
where
    H: BrainstemHardware,
{
    for _ in 0..BOOT_LED_BLINKS {
        hardware.set_indicators(true);
        hardware.delay_ms(120);
        hardware.set_indicators(false);
        hardware.delay_ms(120);
    }
    hardware.set_indicators(true);
    hardware.delay_ms(250);
}
