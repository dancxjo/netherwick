#[derive(Clone, Copy, Eq, PartialEq)]
pub struct ImuSample {
    pub timestamp_ms: u32,
    pub gyro_x_mrad_s: i16,
    pub gyro_y_mrad_s: i16,
    pub gyro_z_mrad_s: i16,
    pub accel_x_mm_s2: i16,
    pub accel_y_mm_s2: i16,
    pub accel_z_mm_s2: i16,
}

impl ImuSample {
    pub const fn stationary(timestamp_ms: u32) -> Self {
        Self {
            timestamp_ms,
            gyro_x_mrad_s: 0,
            gyro_y_mrad_s: 0,
            gyro_z_mrad_s: 0,
            accel_x_mm_s2: 0,
            accel_y_mm_s2: 0,
            accel_z_mm_s2: 9_807,
        }
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum ImuHealth {
    Unknown,
    Ok,
    Fault,
    Absent,
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub struct ImuDerivedState {
    pub yaw_mrad: i32,
    pub pitch_mrad: i16,
    pub roll_mrad: i16,
    pub yaw_rate_mrad_s: i16,
    pub accel_magnitude_mm_s2: u16,
    pub tilt_magnitude_mrad: u16,
    pub roughness_mm_s2: u16,
    pub impact_score_mm_s2: u16,
}

pub trait ImuDriver {
    fn poll(&mut self, now_ms: u32) -> Result<Option<ImuSample>, ImuHealth>;
}

pub trait ImuI2cBus {
    fn write(&mut self, address: u8, bytes: &[u8]) -> Result<(), ()>;
    fn write_read(&mut self, address: u8, bytes: &[u8], read: &mut [u8]) -> Result<(), ()>;
}

impl<T> ImuI2cBus for T
where
    T: embedded_hal::i2c::I2c,
{
    fn write(&mut self, address: u8, bytes: &[u8]) -> Result<(), ()> {
        embedded_hal::i2c::I2c::write(self, address, bytes).map_err(|_| ())
    }

    fn write_read(&mut self, address: u8, bytes: &[u8], read: &mut [u8]) -> Result<(), ()> {
        embedded_hal::i2c::I2c::write_read(self, address, bytes, read).map_err(|_| ())
    }
}

pub struct NoImu;

impl ImuDriver for NoImu {
    fn poll(&mut self, _now_ms: u32) -> Result<Option<ImuSample>, ImuHealth> {
        Ok(None)
    }
}

pub struct Mpu6050<B> {
    bus: B,
    address: u8,
    initialized: bool,
}

impl<B> Mpu6050<B>
where
    B: ImuI2cBus,
{
    pub const DEFAULT_ADDRESS: u8 = 0x68;

    pub const fn new(bus: B) -> Self {
        Self {
            bus,
            address: Self::DEFAULT_ADDRESS,
            initialized: false,
        }
    }

    pub const fn with_address(bus: B, address: u8) -> Self {
        Self {
            bus,
            address,
            initialized: false,
        }
    }

    fn initialize(&mut self) -> Result<(), ImuHealth> {
        let mut who_am_i = [0u8; 1];
        self.bus
            .write_read(self.address, &[Register::WhoAmI as u8], &mut who_am_i)
            .map_err(|_| ImuHealth::Fault)?;
        if who_am_i[0] != 0x68 && who_am_i[0] != 0x70 {
            return Err(ImuHealth::Absent);
        }

        self.bus
            .write(self.address, &[Register::PwrMgmt1 as u8, 0x00])
            .map_err(|_| ImuHealth::Fault)?;
        self.bus
            .write(self.address, &[Register::GyroConfig as u8, 0x00])
            .map_err(|_| ImuHealth::Fault)?;
        self.bus
            .write(self.address, &[Register::AccelConfig as u8, 0x00])
            .map_err(|_| ImuHealth::Fault)?;
        self.initialized = true;
        Ok(())
    }
}

impl<B> ImuDriver for Mpu6050<B>
where
    B: ImuI2cBus,
{
    fn poll(&mut self, now_ms: u32) -> Result<Option<ImuSample>, ImuHealth> {
        if !self.initialized {
            self.initialize()?;
        }

        let mut bytes = [0u8; 14];
        self.bus
            .write_read(self.address, &[Register::AccelXoutH as u8], &mut bytes)
            .map_err(|_| ImuHealth::Fault)?;

        Ok(Some(ImuSample {
            timestamp_ms: now_ms,
            accel_x_mm_s2: accel_raw_to_mm_s2(read_i16(&bytes, 0)),
            accel_y_mm_s2: accel_raw_to_mm_s2(read_i16(&bytes, 2)),
            accel_z_mm_s2: accel_raw_to_mm_s2(read_i16(&bytes, 4)),
            gyro_x_mrad_s: gyro_raw_to_mrad_s(read_i16(&bytes, 8)),
            gyro_y_mrad_s: gyro_raw_to_mrad_s(read_i16(&bytes, 10)),
            gyro_z_mrad_s: gyro_raw_to_mrad_s(read_i16(&bytes, 12)),
        }))
    }
}

#[repr(u8)]
enum Register {
    AccelXoutH = 0x3B,
    GyroConfig = 0x1B,
    AccelConfig = 0x1C,
    PwrMgmt1 = 0x6B,
    WhoAmI = 0x75,
}

fn read_i16(bytes: &[u8; 14], offset: usize) -> i16 {
    i16::from_be_bytes([bytes[offset], bytes[offset + 1]])
}

fn accel_raw_to_mm_s2(raw: i16) -> i16 {
    clamp_i16((raw as i32).saturating_mul(9_807) / 16_384)
}

fn gyro_raw_to_mrad_s(raw: i16) -> i16 {
    clamp_i16((raw as i32).saturating_mul(133) / 1_000)
}

pub fn derive_sample(
    previous_yaw_mrad: i32,
    previous_timestamp_ms: u32,
    previous_accel_magnitude_mm_s2: u16,
    sample: ImuSample,
) -> ImuDerivedState {
    let elapsed_ms = if previous_timestamp_ms == 0 {
        0
    } else {
        sample.timestamp_ms.wrapping_sub(previous_timestamp_ms)
    };
    let yaw_delta = (sample.gyro_z_mrad_s as i32).saturating_mul(elapsed_ms as i32) / 1_000;
    let accel_magnitude = vector_magnitude(
        sample.accel_x_mm_s2 as i32,
        sample.accel_y_mm_s2 as i32,
        sample.accel_z_mm_s2 as i32,
    );
    let pitch = tilt_axis_mrad(sample.accel_x_mm_s2 as i32, sample.accel_z_mm_s2 as i32);
    let roll = tilt_axis_mrad(sample.accel_y_mm_s2 as i32, sample.accel_z_mm_s2 as i32);
    let tilt_magnitude = abs_i32(pitch as i32)
        .max(abs_i32(roll as i32))
        .min(u16::MAX as i32);
    let roughness = if previous_accel_magnitude_mm_s2 == 0 {
        0
    } else {
        abs_i32(accel_magnitude as i32 - previous_accel_magnitude_mm_s2 as i32).min(u16::MAX as i32)
    };

    ImuDerivedState {
        yaw_mrad: previous_yaw_mrad.saturating_add(yaw_delta),
        pitch_mrad: pitch,
        roll_mrad: roll,
        yaw_rate_mrad_s: sample.gyro_z_mrad_s,
        accel_magnitude_mm_s2: accel_magnitude,
        tilt_magnitude_mrad: tilt_magnitude as u16,
        roughness_mm_s2: roughness as u16,
        impact_score_mm_s2: roughness as u16,
    }
}

fn vector_magnitude(x: i32, y: i32, z: i32) -> u16 {
    let square_sum = x
        .saturating_mul(x)
        .saturating_add(y.saturating_mul(y))
        .saturating_add(z.saturating_mul(z)) as u32;
    int_sqrt(square_sum).min(u16::MAX as u32) as u16
}

fn tilt_axis_mrad(axis: i32, vertical: i32) -> i16 {
    if vertical == 0 {
        return 0;
    }
    ((axis.saturating_mul(1_000)) / abs_i32(vertical).max(1))
        .clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

fn int_sqrt(value: u32) -> u32 {
    let mut result = 0u32;
    let mut bit = 1u32 << 30;
    while bit > value {
        bit >>= 2;
    }
    let mut n = value;
    while bit != 0 {
        if n >= result + bit {
            n -= result + bit;
            result = (result >> 1) + bit;
        } else {
            result >>= 1;
        }
        bit >>= 2;
    }
    result
}

fn abs_i32(value: i32) -> i32 {
    if value == i32::MIN {
        i32::MAX
    } else {
        value.abs()
    }
}

fn clamp_i16(value: i32) -> i16 {
    value.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}
