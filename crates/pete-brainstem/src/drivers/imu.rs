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

pub struct NoImu;

impl ImuDriver for NoImu {
    fn poll(&mut self, _now_ms: u32) -> Result<Option<ImuSample>, ImuHealth> {
        Ok(None)
    }
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
    let yaw_delta = (sample.gyro_z_mrad_s as i32)
        .saturating_mul(elapsed_ms as i32)
        / 1_000;
    let accel_magnitude = vector_magnitude(
        sample.accel_x_mm_s2 as i32,
        sample.accel_y_mm_s2 as i32,
        sample.accel_z_mm_s2 as i32,
    );
    let pitch = tilt_axis_mrad(sample.accel_x_mm_s2 as i32, sample.accel_z_mm_s2 as i32);
    let roll = tilt_axis_mrad(sample.accel_y_mm_s2 as i32, sample.accel_z_mm_s2 as i32);
    let tilt_magnitude = abs_i32(pitch as i32).max(abs_i32(roll as i32)).min(u16::MAX as i32);
    let roughness = if previous_accel_magnitude_mm_s2 == 0 {
        0
    } else {
        abs_i32(accel_magnitude as i32 - previous_accel_magnitude_mm_s2 as i32)
            .min(u16::MAX as i32)
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
    ((axis.saturating_mul(1_000)) / abs_i32(vertical).max(1)).clamp(i16::MIN as i32, i16::MAX as i32)
        as i16
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
