use anyhow::Result;
use async_trait::async_trait;
use netherwick_body::{BodySense, MotionCommand, MotorCommand, MotorComplex, RobotBody};
use netherwick_now::{ExtensionSense, GpsSense, ImuSense, RangeSense};
use netherwick_sensors::{EyeFrame, EyeFrameFormat, PcmAudioFrame, World, WorldSnapshot, WorldUpdate};
use rand::{rngs::StdRng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

#[derive(Debug)]
pub struct SimBody {
    body: BodySense,
    _rng: StdRng,
}

impl SimBody {
    pub fn new(seed: u64) -> Self {
        Self {
            body: BodySense::default(),
            _rng: StdRng::seed_from_u64(seed),
        }
    }
}

#[async_trait]
impl RobotBody for SimBody {
    async fn read_body(&mut self) -> Result<BodySense> {
        Ok(self.body.clone())
    }

    async fn apply_motor(&mut self, cmd: MotorCommand) -> Result<()> {
        self.body.velocity.forward_m_s = cmd.forward;
        self.body.velocity.turn_rad_s = cmd.turn;
        self.body.odometry.x_m += cmd.forward * 0.1;
        self.body.odometry.heading_rad += cmd.turn * 0.1;
        self.body.battery_level = (self.body.battery_level - cmd.forward.abs() * 0.01).max(0.0);
        self.body.last_update_ms = self.body.last_update_ms.saturating_add(100);
        Ok(())
    }
}

#[async_trait]
impl MotorComplex for SimBody {
    async fn send(&mut self, command: MotionCommand) -> Result<BodySense> {
        let motor = command.to_motor_command();
        self.apply_motor(motor).await?;
        self.read_body().await
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ArenaConfig {
    pub width_m: f32,
    pub height_m: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SimObject {
    pub label: String,
    pub x_m: f32,
    pub y_m: f32,
    pub radius_m: f32,
    pub color_rgb: [u8; 3],
}

#[derive(Clone, Debug)]
struct VirtualWorldState {
    snapshot: WorldSnapshot,
    arena: ArenaConfig,
    objects: Vec<SimObject>,
}

#[derive(Clone, Debug)]
pub struct VirtualWorld {
    state: Arc<Mutex<VirtualWorldState>>,
}

#[derive(Clone, Debug)]
pub struct SimMotorComplex {
    state: Arc<Mutex<VirtualWorldState>>,
}

impl VirtualWorld {
    pub fn new(seed: u64, arena: ArenaConfig) -> Self {
        let (world, _) = Self::new_with_motor(seed, arena);
        world
    }

    pub fn new_with_motor(seed: u64, arena: ArenaConfig) -> (Self, SimMotorComplex) {
        let mut snapshot = WorldSnapshot::default();
        snapshot.body.last_update_ms = seed;
        let state = Arc::new(Mutex::new(VirtualWorldState {
            snapshot,
            arena,
            objects: Vec::new(),
        }));
        (
            Self {
                state: Arc::clone(&state),
            },
            SimMotorComplex { state },
        )
    }

    pub fn add_object(&mut self, object: SimObject) {
        self.state
            .lock()
            .expect("virtual world mutex poisoned")
            .objects
            .push(object);
    }

    pub fn set_objects(&mut self, objects: Vec<SimObject>) {
        self.state
            .lock()
            .expect("virtual world mutex poisoned")
            .objects = objects;
    }

    fn project_snapshot(state: &mut VirtualWorldState) {
        let body = &state.snapshot.body;
        state.snapshot.range = RangeSense {
            schema_version: 1,
            beams: project_range_beams(body, &state.objects, state.arena),
            nearest_m: nearest_object_distance(body, &state.objects),
        };
        state.snapshot.imu = ImuSense {
            schema_version: 1,
            orientation: vec![body.odometry.heading_rad],
            acceleration: vec![body.velocity.forward_m_s, 0.0, 0.0],
            angular_velocity: vec![0.0, 0.0, body.velocity.turn_rad_s],
        };
        state.snapshot.gps = Some(GpsSense {
            schema_version: 1,
            lat: 37.0 + body.odometry.y_m as f64 / 111_111.0,
            lon: -122.0 + body.odometry.x_m as f64 / 111_111.0,
            altitude_m: Some(0.0),
        });
        state.snapshot.eye_frame = Some(project_eye_frame(body, &state.objects, state.arena));
        state.snapshot.eye.frames = state
            .snapshot
            .eye_frame
            .as_ref()
            .map(|frame| {
                frame
                    .bytes
                    .chunks(3)
                    .take(256)
                    .map(|pixel| pixel.iter().copied().map(|value| value as f32 / 255.0).sum())
                    .collect::<Vec<f32>>()
            })
            .into_iter()
            .collect();
        if let Some(ear_pcm) = &state.snapshot.ear_pcm {
            state.snapshot.ear.features = vec![ear_pcm
                .samples
                .iter()
                .take(256)
                .map(|sample| *sample as f32 / i16::MAX as f32)
                .collect()];
        }
        state.snapshot.extensions = vec![ExtensionSense {
            schema_version: 1,
            name: "sim.world".to_string(),
            values: vec![
                state.arena.width_m,
                state.arena.height_m,
                state.objects.len() as f32,
            ],
        }];
    }
}

#[async_trait]
impl World for VirtualWorld {
    async fn snapshot(&mut self) -> Result<WorldSnapshot> {
        let mut guard = self.state.lock().expect("virtual world mutex poisoned");
        Self::project_snapshot(&mut guard);
        Ok(guard.snapshot.clone())
    }

    async fn apply_update(&mut self, update: WorldUpdate) -> Result<()> {
        let mut guard = self.state.lock().expect("virtual world mutex poisoned");
        update.apply_to(&mut guard.snapshot);
        Ok(())
    }
}

#[async_trait]
impl MotorComplex for SimMotorComplex {
    async fn send(&mut self, command: MotionCommand) -> Result<BodySense> {
        let motor = command.to_motor_command();
        let mut guard = self.state.lock().expect("virtual world mutex poisoned");
        let body = &mut guard.snapshot.body;
        body.velocity.forward_m_s = motor.forward;
        body.velocity.turn_rad_s = motor.turn;
        body.odometry.heading_rad += motor.turn * 0.1;
        body.odometry.x_m += motor.forward * body.odometry.heading_rad.cos() * 0.1;
        body.odometry.y_m += motor.forward * body.odometry.heading_rad.sin() * 0.1;
        body.battery_level = (body.battery_level - (motor.forward.abs() + motor.turn.abs()) * 0.005)
            .clamp(0.0, 1.0);
        body.last_update_ms = body.last_update_ms.saturating_add(100);
        Ok(body.clone())
    }
}

#[async_trait]
impl RobotBody for SimMotorComplex {
    async fn read_body(&mut self) -> Result<BodySense> {
        Ok(self
            .state
            .lock()
            .expect("virtual world mutex poisoned")
            .snapshot
            .body
            .clone())
    }

    async fn apply_motor(&mut self, cmd: MotorCommand) -> Result<()> {
        self.send(MotionCommand::Drive {
            forward_m_s: cmd.forward,
            turn_rad_s: cmd.turn,
        })
        .await?;
        Ok(())
    }
}

fn project_eye_frame(body: &BodySense, objects: &[SimObject], arena: ArenaConfig) -> EyeFrame {
    let width = 64usize;
    let height = 48usize;
    let mut bytes = vec![20u8; width * height * 3];
    let heading = body.odometry.heading_rad;
    for object in objects {
        let dx = object.x_m - body.odometry.x_m;
        let dy = object.y_m - body.odometry.y_m;
        let local_x = dx * heading.cos() + dy * heading.sin();
        let local_y = -dx * heading.sin() + dy * heading.cos();
        if local_x <= 0.0 {
            continue;
        }
        let screen_x = ((local_y / arena.width_m) + 0.5) * width as f32;
        let screen_y = (1.0 - (local_x / arena.height_m)).clamp(0.0, 1.0) * height as f32;
        let radius = (object.radius_m * 12.0).max(1.0) as i32;
        draw_disc(
            &mut bytes,
            width,
            height,
            screen_x as i32,
            screen_y as i32,
            radius,
            object.color_rgb,
        );
    }
    EyeFrame {
        captured_at_ms: body.last_update_ms,
        width: width as u32,
        height: height as u32,
        format: EyeFrameFormat::Rgb8,
        bytes,
    }
}

fn draw_disc(
    bytes: &mut [u8],
    width: usize,
    height: usize,
    center_x: i32,
    center_y: i32,
    radius: i32,
    color: [u8; 3],
) {
    for y in (center_y - radius).max(0)..=(center_y + radius).min(height as i32 - 1) {
        for x in (center_x - radius).max(0)..=(center_x + radius).min(width as i32 - 1) {
            let dx = x - center_x;
            let dy = y - center_y;
            if dx * dx + dy * dy > radius * radius {
                continue;
            }
            let index = ((y as usize * width) + x as usize) * 3;
            bytes[index..index + 3].copy_from_slice(&color);
        }
    }
}

fn project_range_beams(body: &BodySense, objects: &[SimObject], arena: ArenaConfig) -> Vec<f32> {
    let max_range = arena.width_m.max(arena.height_m).max(1.0);
    let nearest = nearest_object_distance(body, objects).unwrap_or(max_range);
    vec![nearest / max_range; 8]
}

fn nearest_object_distance(body: &BodySense, objects: &[SimObject]) -> Option<f32> {
    objects
        .iter()
        .map(|object| {
            let dx = object.x_m - body.odometry.x_m;
            let dy = object.y_m - body.odometry.y_m;
            ((dx * dx) + (dy * dy)).sqrt().max(0.0)
        })
        .min_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal))
}
