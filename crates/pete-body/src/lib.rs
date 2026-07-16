use pete_core::{Pose2, TimeMs};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BodyTone {
    pub note: u8,
    pub duration_64ths: u8,
}

impl BodyTone {
    pub fn new(note: u8, duration_64ths: u8) -> Self {
        Self {
            note,
            duration_64ths,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BodySong {
    pub tones: Vec<BodyTone>,
}

impl BodySong {
    pub fn new(tones: impl Into<Vec<BodyTone>>) -> Self {
        Self {
            tones: tones.into(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BodyFlags {
    pub bump_left: bool,
    pub bump_right: bool,
    pub cliff_left: bool,
    pub cliff_front_left: bool,
    pub cliff_front_right: bool,
    pub cliff_right: bool,
    pub wheel_drop: bool,
    pub wall: bool,
    pub virtual_wall: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CliffSensors {
    pub left: f32,
    pub front_left: f32,
    pub front_right: f32,
    pub right: f32,
}

impl CliffSensors {
    pub fn max(self) -> f32 {
        self.left
            .max(self.front_left)
            .max(self.front_right)
            .max(self.right)
            .clamp(0.0, 1.0)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Velocity {
    pub forward_m_s: f32,
    pub turn_rad_s: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BodyHealth {
    pub strain: f32,
    pub health: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ImuVector3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl ImuVector3 {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn magnitude(self) -> f32 {
        (self
            .x
            .mul_add(self.x, self.y.mul_add(self.y, self.z * self.z)))
        .sqrt()
    }

    pub fn dot(self, other: Self) -> f32 {
        self.x
            .mul_add(other.x, self.y.mul_add(other.y, self.z * other.z))
    }

    pub fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImuGravityCalibration {
    /// Stationary accelerometer vector that should count as "level".
    ///
    /// Accelerometers measure specific force, so this points opposite physical
    /// down while the robot is still. `down()` returns the physical down vector.
    pub level_acceleration: ImuVector3,
    pub level_magnitude: f32,
}

impl ImuGravityCalibration {
    pub fn zero_from_stationary_acceleration(acceleration: ImuVector3) -> Option<Self> {
        let level_magnitude = acceleration.magnitude();
        if !level_magnitude.is_finite() || level_magnitude < 0.1 {
            return None;
        }
        Some(Self {
            level_acceleration: acceleration,
            level_magnitude,
        })
    }

    pub fn down(self) -> ImuVector3 {
        ImuVector3::new(
            -self.level_acceleration.x,
            -self.level_acceleration.y,
            -self.level_acceleration.z,
        )
    }

    pub fn calibrated_tilt_rad(self, acceleration: ImuVector3) -> Option<f32> {
        let current_magnitude = acceleration.magnitude();
        if !current_magnitude.is_finite() || current_magnitude < 0.1 {
            return None;
        }
        let cross = self.level_acceleration.cross(acceleration).magnitude();
        let dot = self.level_acceleration.dot(acceleration);
        Some(cross.atan2(dot).abs())
    }

    pub fn is_level_within(self, acceleration: ImuVector3, tolerance_rad: f32) -> bool {
        self.calibrated_tilt_rad(acceleration)
            .is_some_and(|tilt| tilt <= tolerance_rad)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BodySense {
    pub battery_level: f32,
    pub charging: bool,
    /// Raw character reported by the Create's omnidirectional IR receiver.
    ///
    /// Zero means that no IR character is currently being received. Non-zero
    /// values retain the Create OI byte so dock and remote-control signals can
    /// be interpreted by higher-level consumers without losing information.
    #[serde(default)]
    pub infrared_character: u8,
    #[serde(default)]
    pub cliff_sensors: CliffSensors,
    pub flags: BodyFlags,
    pub odometry: Pose2,
    pub velocity: Velocity,
    pub health: BodyHealth,
    pub last_update_ms: TimeMs,
}

impl Default for BodySense {
    fn default() -> Self {
        Self {
            battery_level: 1.0,
            charging: false,
            infrared_character: 0,
            cliff_sensors: CliffSensors::default(),
            flags: BodyFlags::default(),
            odometry: Pose2::default(),
            velocity: Velocity::default(),
            health: BodyHealth {
                strain: 0.0,
                health: 1.0,
            },
            last_update_ms: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gravity_calibration_detects_down_for_arbitrary_mount() {
        let calibration = ImuGravityCalibration::zero_from_stationary_acceleration(
            ImuVector3::new(1.0, 0.0, 0.0),
        )
        .unwrap();

        assert_eq!(calibration.down(), ImuVector3::new(-1.0, -0.0, -0.0));
        assert!(calibration.is_level_within(ImuVector3::new(1.0, 0.0, 0.0), 0.001));
    }

    #[test]
    fn gravity_calibration_reports_tilt_from_zeroed_direction() {
        let calibration = ImuGravityCalibration::zero_from_stationary_acceleration(
            ImuVector3::new(1.0, 0.0, 0.0),
        )
        .unwrap();

        let tilt = calibration
            .calibrated_tilt_rad(ImuVector3::new(0.0, 1.0, 0.0))
            .unwrap();

        assert!((tilt - core::f32::consts::FRAC_PI_2).abs() < 0.001);
    }
}
