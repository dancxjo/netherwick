use anyhow::Result;
use async_trait::async_trait;
use netherwick_body::{
    BodySense, CliffSensors, MotionCommand, MotorCommand, MotorComplex, RobotBody,
};
use netherwick_now::{
    EarSense, ExtensionSense, FaceSense, GpsSense, ImuSense, KinectJointSense, KinectSense,
    KinectSkeletonSense, RangeSense, VoiceSense,
};
use netherwick_sensors::{
    EyeFrame, EyeFrameFormat, PcmAudioFrame, World, WorldSnapshot, WorldUpdate,
};
use rand::{rngs::StdRng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

pub mod scenario;
pub use scenario::{
    build_scenario, default_sim_world, ScenarioConfig, ScenarioKind, ScenarioMetadata,
    ScenarioWorld,
};

const ROBOT_RADIUS_M: f32 = 0.18;
const SIM_DT_S: f32 = 0.1;
const RANGE_BEAM_COUNT: usize = 8;
const RANGE_FOV_RAD: f32 = std::f32::consts::PI;
const RANGE_MAX_M: f32 = 4.0;
const VISIBLE_FOV_RAD: f32 = std::f32::consts::FRAC_PI_2;
const VISIBLE_MAX_M: f32 = 4.0;
const EYE_WIDTH: usize = 160;
const EYE_HEIGHT: usize = 90;
const EYE_HORIZONTAL_FOV_RAD: f32 = 70.0_f32.to_radians();
const EYE_CAMERA_HEIGHT_M: f32 = 0.25;
const EYE_NEAR_PLANE_M: f32 = 0.05;

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
pub enum SimObjectKind {
    Obstacle,
    Charger,
    Person { identity: Option<String> },
    SoundSource { label: String },
    Landmark { label: String },
}

impl Default for SimObjectKind {
    fn default() -> Self {
        Self::Obstacle
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SimObject {
    pub id: String,
    pub label: String,
    pub kind: SimObjectKind,
    pub x_m: f32,
    pub y_m: f32,
    pub radius_m: f32,
    pub color_rgb: [u8; 3],
    pub emits_sound: bool,
    pub charge_rate: f32,
}

impl SimObject {
    pub fn obstacle(
        id: impl Into<String>,
        label: impl Into<String>,
        x_m: f32,
        y_m: f32,
        radius_m: f32,
    ) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            kind: SimObjectKind::Obstacle,
            x_m,
            y_m,
            radius_m,
            color_rgb: [180, 90, 80],
            emits_sound: false,
            charge_rate: 0.0,
        }
    }

    pub fn charger(
        id: impl Into<String>,
        label: impl Into<String>,
        x_m: f32,
        y_m: f32,
        radius_m: f32,
    ) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            kind: SimObjectKind::Charger,
            x_m,
            y_m,
            radius_m,
            color_rgb: [80, 220, 130],
            emits_sound: false,
            charge_rate: 0.25,
        }
    }
}

#[derive(Clone, Debug)]
struct VirtualWorldState {
    snapshot: WorldSnapshot,
    arena: ArenaConfig,
    objects: Vec<SimObject>,
    retina_frame: Option<EyeFrame>,
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
        snapshot.body.odometry.x_m = arena.width_m * 0.5;
        snapshot.body.odometry.y_m = arena.height_m * 0.5;
        let state = Arc::new(Mutex::new(VirtualWorldState {
            snapshot,
            arena,
            objects: Vec::new(),
            retina_frame: None,
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

    pub fn set_retina_frame(&mut self, frame: Option<EyeFrame>) {
        let mut guard = self.state.lock().expect("virtual world mutex poisoned");
        guard.retina_frame = frame;
    }

    pub fn set_objects(&mut self, objects: Vec<SimObject>) {
        self.state
            .lock()
            .expect("virtual world mutex poisoned")
            .objects = objects;
    }

    pub fn set_body(&mut self, body: BodySense) {
        self.state
            .lock()
            .expect("virtual world mutex poisoned")
            .snapshot
            .body = body;
    }

    pub fn arena(&self) -> ArenaConfig {
        self.state
            .lock()
            .expect("virtual world mutex poisoned")
            .arena
    }

    pub fn body(&self) -> BodySense {
        self.state
            .lock()
            .expect("virtual world mutex poisoned")
            .snapshot
            .body
            .clone()
    }

    pub fn reset_body_to_spawn(&mut self) -> BodySense {
        let mut guard = self.state.lock().expect("virtual world mutex poisoned");
        let previous_update_ms = guard.snapshot.body.last_update_ms;
        let mut body = BodySense::default();
        body.last_update_ms = previous_update_ms.saturating_add(100);
        body.odometry.x_m = guard.arena.width_m * 0.5;
        body.odometry.y_m = guard.arena.height_m * 0.5;
        guard.snapshot.body = body.clone();
        Self::project_snapshot(&mut guard);
        body
    }

    pub fn objects(&self) -> Vec<SimObject> {
        self.state
            .lock()
            .expect("virtual world mutex poisoned")
            .objects
            .clone()
    }

    fn project_snapshot(state: &mut VirtualWorldState) {
        apply_cliff_projection(&mut state.snapshot.body, state.arena);
        let body = &state.snapshot.body;
        state.snapshot.range = RangeSense {
            schema_version: 1,
            beams: project_range_beams(
                body,
                &state.objects,
                state.arena,
                RANGE_BEAM_COUNT,
                RANGE_FOV_RAD,
                RANGE_MAX_M,
            ),
            nearest_m: nearest_obstacle_distance(body, &state.objects, state.arena, RANGE_MAX_M),
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
        if let Some(ref retina) = state.retina_frame {
            state.snapshot.eye_frame = Some(retina.clone());
        } else {
            let mut fallback = project_eye_frame(body, &state.objects, state.arena);
            fallback.source = Some("rust-sim-symbolic".to_string());
            state.snapshot.eye_frame = Some(fallback);
        }
        state.snapshot.eye.frames = state
            .snapshot
            .eye_frame
            .as_ref()
            .map(|frame| {
                frame
                    .bytes
                    .chunks(3)
                    .take(256)
                    .map(|pixel| {
                        pixel
                            .iter()
                            .copied()
                            .map(|value| value as f32 / 255.0)
                            .sum()
                    })
                    .collect::<Vec<f32>>()
            })
            .into_iter()
            .collect();
        state.snapshot.face = project_face_sense(body, &state.objects);
        state.snapshot.voice = project_voice_sense(body, &state.objects);
        let (ear, ear_pcm) = project_ear_sense(body, &state.objects);
        state.snapshot.ear = ear;
        state.snapshot.ear_pcm = ear_pcm;
        state.snapshot.kinect = project_kinect_sense(body, &state.objects, state.arena);
        state.snapshot.extensions = vec![ExtensionSense {
            schema_version: 1,
            name: "sim.world".to_string(),
            values: vec![
                state.arena.width_m,
                state.arena.height_m,
                state.objects.len() as f32,
                charger_near_score(body, &state.objects),
                charger_visible_score(body, &state.objects),
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
        let objects = guard.objects.clone();
        let arena = guard.arena;
        let body = &mut guard.snapshot.body;
        body.velocity.forward_m_s = motor.forward;
        body.velocity.turn_rad_s = motor.turn;
        body.odometry.heading_rad += motor.turn * 0.1;
        body.flags.bump_left = false;
        body.flags.bump_right = false;
        body.flags.cliff_left = false;
        body.flags.cliff_front_left = false;
        body.flags.cliff_front_right = false;
        body.flags.cliff_right = false;
        body.flags.wall = false;

        let proposed_x =
            body.odometry.x_m + motor.forward * body.odometry.heading_rad.cos() * SIM_DT_S;
        let proposed_y =
            body.odometry.y_m + motor.forward * body.odometry.heading_rad.sin() * SIM_DT_S;
        let collision = collision_at(proposed_x, proposed_y, &objects, arena);
        if collision.collided {
            body.velocity.forward_m_s = 0.0;
            body.flags.bump_left = collision.bump_left;
            body.flags.bump_right = collision.bump_right;
            body.flags.wall = collision.wall;
        } else {
            body.odometry.x_m = proposed_x;
            body.odometry.y_m = proposed_y;
        }

        let charge_rate = charger_contact_rate(body, &objects);
        body.charging = charge_rate > 0.0;
        body.battery_level =
            (body.battery_level - (motor.forward.abs() + motor.turn.abs()) * 0.005).clamp(0.0, 1.0);
        if body.charging {
            body.battery_level = (body.battery_level + charge_rate * SIM_DT_S).clamp(0.0, 1.0);
        }
        apply_cliff_projection(body, arena);
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
    let width = EYE_WIDTH;
    let height = EYE_HEIGHT;
    let mut bytes = vec![0u8; width * height * 3];
    let mut depth = vec![f32::INFINITY; width * height];
    let focal_x = width as f32 * 0.5 / (EYE_HORIZONTAL_FOV_RAD * 0.5).tan();
    let focal_y = focal_x;

    draw_eye_background(&mut bytes, width, height);
    draw_eye_walls(
        &mut bytes, &mut depth, width, height, focal_x, focal_y, body, arena,
    );

    let heading = body.odometry.heading_rad;
    let mut visible = objects
        .iter()
        .filter_map(|object| {
            let dx = object.x_m - body.odometry.x_m;
            let dy = object.y_m - body.odometry.y_m;
            let forward = dx * heading.cos() + dy * heading.sin();
            let lateral = -dx * heading.sin() + dy * heading.cos();
            (forward > EYE_NEAR_PLANE_M).then_some((object, forward, lateral))
        })
        .filter(|(object, forward, lateral)| {
            let angular_half_width = (object.radius_m / *forward).atan();
            let angle = lateral.atan2(*forward);
            angle.abs() - angular_half_width <= EYE_HORIZONTAL_FOV_RAD * 0.5
        })
        .collect::<Vec<_>>();
    visible.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for (object, forward, lateral) in visible {
        draw_projected_object(
            &mut bytes, &mut depth, width, height, focal_x, focal_y, object, forward, lateral,
        );
    }

    draw_reticle(&mut bytes, width, height);

    EyeFrame {
        captured_at_ms: body.last_update_ms,
        width: width as u32,
        height: height as u32,
        format: EyeFrameFormat::Rgb8,
        bytes,
        source: Some("rust-sim-symbolic".to_string()),
    }
}

fn draw_eye_background(bytes: &mut [u8], width: usize, height: usize) {
    let horizon = (height as f32 * 0.48) as usize;
    for y in 0..height {
        let color = if y < horizon {
            let t = y as f32 / horizon.max(1) as f32;
            lerp_color([33, 45, 58], [72, 84, 94], t)
        } else {
            let t = (y - horizon) as f32 / (height - horizon).max(1) as f32;
            lerp_color([69, 72, 70], [39, 43, 42], t)
        };
        for x in 0..width {
            set_pixel(bytes, width, x, y, color);
        }
    }
    for x in 0..width {
        set_pixel(bytes, width, x, horizon.min(height - 1), [106, 111, 112]);
    }
}

fn draw_eye_walls(
    bytes: &mut [u8],
    depth: &mut [f32],
    width: usize,
    height: usize,
    focal_x: f32,
    focal_y: f32,
    body: &BodySense,
    arena: ArenaConfig,
) {
    let samples = ((arena.width_m.max(arena.height_m) / 0.08).ceil() as usize).max(16);
    for index in 0..=samples {
        let t = index as f32 / samples as f32;
        let x = arena.width_m * t;
        draw_wall_sample(bytes, depth, width, height, focal_x, focal_y, body, x, 0.0);
        draw_wall_sample(
            bytes,
            depth,
            width,
            height,
            focal_x,
            focal_y,
            body,
            x,
            arena.height_m,
        );
        let y = arena.height_m * t;
        draw_wall_sample(bytes, depth, width, height, focal_x, focal_y, body, 0.0, y);
        draw_wall_sample(
            bytes,
            depth,
            width,
            height,
            focal_x,
            focal_y,
            body,
            arena.width_m,
            y,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_wall_sample(
    bytes: &mut [u8],
    depth: &mut [f32],
    width: usize,
    height: usize,
    focal_x: f32,
    focal_y: f32,
    body: &BodySense,
    world_x: f32,
    world_y: f32,
) {
    let Some((screen_x, floor_y, forward)) =
        project_world_point(body, world_x, world_y, 0.0, width, height, focal_x, focal_y)
    else {
        return;
    };
    let Some((_, top_y, _)) = project_world_point(
        body, world_x, world_y, 0.65, width, height, focal_x, focal_y,
    ) else {
        return;
    };
    if screen_x < -2.0 || screen_x > width as f32 + 2.0 {
        return;
    }
    let x = screen_x.round() as i32;
    let y0 = top_y.round().min(floor_y.round()) as i32;
    let y1 = top_y.round().max(floor_y.round()) as i32;
    let shade = (165.0 - forward * 18.0).clamp(72.0, 150.0) as u8;
    for dx in -1..=1 {
        draw_depth_line(
            bytes,
            depth,
            width,
            height,
            x + dx,
            y0,
            x + dx,
            y1,
            forward,
            [shade, shade.saturating_add(6), shade.saturating_add(10)],
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_projected_object(
    bytes: &mut [u8],
    depth: &mut [f32],
    width: usize,
    height: usize,
    focal_x: f32,
    focal_y: f32,
    object: &SimObject,
    forward: f32,
    lateral: f32,
) {
    let center_x = focal_x * lateral / forward + width as f32 * 0.5;
    let bottom_y = height as f32 * 0.5 + focal_y * EYE_CAMERA_HEIGHT_M / forward;
    let visual_height = object_visual_height(object);
    let top_y = height as f32 * 0.5 - focal_y * (visual_height - EYE_CAMERA_HEIGHT_M) / forward;
    let half_width = (focal_x * object.radius_m / forward).max(1.5);
    let min_x = (center_x - half_width).floor().max(0.0) as i32;
    let max_x = (center_x + half_width).ceil().min(width as f32 - 1.0) as i32;
    let min_y = top_y.floor().max(0.0) as i32;
    let max_y = bottom_y.ceil().min(height as f32 - 1.0) as i32;
    if min_x > max_x || min_y > max_y {
        return;
    }
    let color = shade_object_color(object.color_rgb, forward);
    match object.kind {
        SimObjectKind::Charger => {
            let pad_top = (bottom_y - (bottom_y - top_y).abs().max(3.0)).max(0.0) as i32;
            draw_depth_ellipse(
                bytes,
                depth,
                width,
                height,
                center_x,
                (pad_top as f32 + bottom_y) * 0.5,
                half_width * 1.35,
                ((bottom_y - pad_top as f32) * 0.5).max(2.0),
                forward,
                color,
            );
        }
        SimObjectKind::Person { .. } => draw_depth_capsule(
            bytes, depth, width, height, min_x, max_x, min_y, max_y, forward, color,
        ),
        SimObjectKind::SoundSource { .. } => draw_depth_triangle(
            bytes, depth, width, height, center_x, min_y, max_y, half_width, forward, color,
        ),
        SimObjectKind::Obstacle | SimObjectKind::Landmark { .. } => draw_depth_rectangle(
            bytes, depth, width, height, min_x, max_x, min_y, max_y, forward, color,
        ),
    }
}

fn object_visual_height(object: &SimObject) -> f32 {
    match object.kind {
        SimObjectKind::Obstacle => 0.45,
        SimObjectKind::Charger => 0.08,
        SimObjectKind::Person { .. } => 1.2,
        SimObjectKind::SoundSource { .. } => 0.35,
        SimObjectKind::Landmark { .. } => 0.6,
    }
}

#[allow(clippy::too_many_arguments)]
fn project_world_point(
    body: &BodySense,
    world_x: f32,
    world_y: f32,
    world_z: f32,
    width: usize,
    height: usize,
    focal_x: f32,
    focal_y: f32,
) -> Option<(f32, f32, f32)> {
    let heading = body.odometry.heading_rad;
    let dx = world_x - body.odometry.x_m;
    let dy = world_y - body.odometry.y_m;
    let forward = dx * heading.cos() + dy * heading.sin();
    if forward <= EYE_NEAR_PLANE_M {
        return None;
    }
    let lateral = -dx * heading.sin() + dy * heading.cos();
    Some((
        focal_x * lateral / forward + width as f32 * 0.5,
        height as f32 * 0.5 - focal_y * (world_z - EYE_CAMERA_HEIGHT_M) / forward,
        forward,
    ))
}

fn shade_object_color(color: [u8; 3], forward: f32) -> [u8; 3] {
    let light = (1.0 - forward / 8.0).clamp(0.45, 1.0);
    [
        (color[0] as f32 * light) as u8,
        (color[1] as f32 * light) as u8,
        (color[2] as f32 * light) as u8,
    ]
}

#[allow(clippy::too_many_arguments)]
fn draw_depth_rectangle(
    bytes: &mut [u8],
    depth: &mut [f32],
    width: usize,
    height: usize,
    min_x: i32,
    max_x: i32,
    min_y: i32,
    max_y: i32,
    z: f32,
    color: [u8; 3],
) {
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            put_depth_pixel(bytes, depth, width, height, x, y, z, color);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_depth_capsule(
    bytes: &mut [u8],
    depth: &mut [f32],
    width: usize,
    height: usize,
    min_x: i32,
    max_x: i32,
    min_y: i32,
    max_y: i32,
    z: f32,
    color: [u8; 3],
) {
    let radius = ((max_x - min_x) as f32 * 0.5).max(1.0);
    let center_x = (min_x + max_x) as f32 * 0.5;
    let cap_center_y = min_y as f32 + radius;
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let dx = x as f32 - center_x;
            let inside = if y as f32 <= cap_center_y {
                let dy = y as f32 - cap_center_y;
                dx * dx + dy * dy <= radius * radius
            } else {
                dx.abs() <= radius
            };
            if inside {
                put_depth_pixel(bytes, depth, width, height, x, y, z, color);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_depth_ellipse(
    bytes: &mut [u8],
    depth: &mut [f32],
    width: usize,
    height: usize,
    center_x: f32,
    center_y: f32,
    radius_x: f32,
    radius_y: f32,
    z: f32,
    color: [u8; 3],
) {
    let min_x = (center_x - radius_x).floor().max(0.0) as i32;
    let max_x = (center_x + radius_x).ceil().min(width as f32 - 1.0) as i32;
    let min_y = (center_y - radius_y).floor().max(0.0) as i32;
    let max_y = (center_y + radius_y).ceil().min(height as f32 - 1.0) as i32;
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let dx = (x as f32 - center_x) / radius_x.max(1.0);
            let dy = (y as f32 - center_y) / radius_y.max(1.0);
            if dx * dx + dy * dy <= 1.0 {
                put_depth_pixel(bytes, depth, width, height, x, y, z, color);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_depth_triangle(
    bytes: &mut [u8],
    depth: &mut [f32],
    width: usize,
    height: usize,
    center_x: f32,
    min_y: i32,
    max_y: i32,
    half_width: f32,
    z: f32,
    color: [u8; 3],
) {
    let span_y = (max_y - min_y).max(1) as f32;
    for y in min_y..=max_y {
        let t = (y - min_y) as f32 / span_y;
        let row_half_width = half_width * (0.25 + t * 0.9);
        for x in
            (center_x - row_half_width).floor() as i32..=(center_x + row_half_width).ceil() as i32
        {
            put_depth_pixel(bytes, depth, width, height, x, y, z, color);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_depth_line(
    bytes: &mut [u8],
    depth: &mut [f32],
    width: usize,
    height: usize,
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    z: f32,
    color: [u8; 3],
) {
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let mut x = x0;
    let mut y = y0;
    loop {
        put_depth_pixel(bytes, depth, width, height, x, y, z, color);
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
}

fn draw_reticle(bytes: &mut [u8], width: usize, height: usize) {
    let center_x = width as i32 / 2;
    let center_y = height as i32 / 2;
    for offset in -4i32..=4 {
        if offset.abs() > 1 {
            put_pixel_i32(
                bytes,
                width,
                height,
                center_x + offset,
                center_y,
                [220, 226, 214],
            );
            put_pixel_i32(
                bytes,
                width,
                height,
                center_x,
                center_y + offset,
                [220, 226, 214],
            );
        }
    }
}

fn put_depth_pixel(
    bytes: &mut [u8],
    depth: &mut [f32],
    width: usize,
    height: usize,
    x: i32,
    y: i32,
    z: f32,
    color: [u8; 3],
) {
    if x < 0 || y < 0 || x >= width as i32 || y >= height as i32 {
        return;
    }
    let index = y as usize * width + x as usize;
    if z <= depth[index] {
        depth[index] = z;
        bytes[index * 3..index * 3 + 3].copy_from_slice(&color);
    }
}

fn put_pixel_i32(bytes: &mut [u8], width: usize, height: usize, x: i32, y: i32, color: [u8; 3]) {
    if x >= 0 && y >= 0 && x < width as i32 && y < height as i32 {
        set_pixel(bytes, width, x as usize, y as usize, color);
    }
}

fn set_pixel(bytes: &mut [u8], width: usize, x: usize, y: usize, color: [u8; 3]) {
    let index = (y * width + x) * 3;
    bytes[index..index + 3].copy_from_slice(&color);
}

fn lerp_color(a: [u8; 3], b: [u8; 3], t: f32) -> [u8; 3] {
    [
        (a[0] as f32 + (b[0] as f32 - a[0] as f32) * t) as u8,
        (a[1] as f32 + (b[1] as f32 - a[1] as f32) * t) as u8,
        (a[2] as f32 + (b[2] as f32 - a[2] as f32) * t) as u8,
    ]
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

#[allow(dead_code)]
fn project_symbolic_eye_frame(
    body: &BodySense,
    objects: &[SimObject],
    arena: ArenaConfig,
) -> EyeFrame {
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
        source: Some("rust-sim-symbolic".to_string()),
    }
}

pub fn project_range_beams(
    body: &BodySense,
    objects: &[SimObject],
    arena: ArenaConfig,
    beam_count: usize,
    fov_rad: f32,
    max_range_m: f32,
) -> Vec<f32> {
    let beam_count = beam_count.max(1);
    let start = if beam_count == 1 { 0.0 } else { -fov_rad * 0.5 };
    let step = if beam_count == 1 {
        0.0
    } else {
        fov_rad / (beam_count - 1) as f32
    };
    (0..beam_count)
        .map(|index| {
            let angle = body.odometry.heading_rad + start + step * index as f32;
            let distance = raycast_distance(
                body.odometry.x_m,
                body.odometry.y_m,
                angle.cos(),
                angle.sin(),
                objects,
                arena,
                max_range_m,
            );
            (distance / max_range_m).clamp(0.0, 1.0)
        })
        .collect()
}

fn nearest_obstacle_distance(
    body: &BodySense,
    objects: &[SimObject],
    arena: ArenaConfig,
    max_range_m: f32,
) -> Option<f32> {
    let wall = raycast_distance(
        body.odometry.x_m,
        body.odometry.y_m,
        body.odometry.heading_rad.cos(),
        body.odometry.heading_rad.sin(),
        objects,
        arena,
        max_range_m,
    );
    let object = objects
        .iter()
        .filter(|object| blocks_motion(object))
        .map(|object| {
            let dx = object.x_m - body.odometry.x_m;
            let dy = object.y_m - body.odometry.y_m;
            ((dx * dx) + (dy * dy)).sqrt() - object.radius_m - ROBOT_RADIUS_M
        })
        .min_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    Some(object.unwrap_or(max_range_m).min(wall).max(0.0))
}

#[derive(Clone, Copy, Debug, Default)]
struct Collision {
    collided: bool,
    bump_left: bool,
    bump_right: bool,
    wall: bool,
}

fn collision_at(x_m: f32, y_m: f32, objects: &[SimObject], arena: ArenaConfig) -> Collision {
    let mut collision = Collision::default();
    if x_m - ROBOT_RADIUS_M < 0.0
        || y_m - ROBOT_RADIUS_M < 0.0
        || x_m + ROBOT_RADIUS_M > arena.width_m
        || y_m + ROBOT_RADIUS_M > arena.height_m
    {
        collision.collided = true;
        collision.wall = true;
        collision.bump_left = true;
        collision.bump_right = true;
        return collision;
    }

    for object in objects.iter().filter(|object| blocks_motion(object)) {
        let dx = object.x_m - x_m;
        let dy = object.y_m - y_m;
        if (dx * dx + dy * dy).sqrt() < object.radius_m + ROBOT_RADIUS_M {
            collision.collided = true;
            collision.bump_left = dy >= 0.0;
            collision.bump_right = dy <= 0.0;
            if !collision.bump_left && !collision.bump_right {
                collision.bump_left = true;
                collision.bump_right = true;
            }
            return collision;
        }
    }

    collision
}

fn blocks_motion(object: &SimObject) -> bool {
    matches!(
        object.kind,
        SimObjectKind::Obstacle | SimObjectKind::Person { .. } | SimObjectKind::Landmark { .. }
    )
}

fn apply_cliff_projection(body: &mut BodySense, arena: ArenaConfig) {
    let sensors = project_cliff_sensors(body, arena);
    body.cliff_sensors = sensors;
    body.flags.cliff_left = sensors.left >= 0.5;
    body.flags.cliff_front_left = sensors.front_left >= 0.5;
    body.flags.cliff_front_right = sensors.front_right >= 0.5;
    body.flags.cliff_right = sensors.right >= 0.5;
}

fn project_cliff_sensors(body: &BodySense, arena: ArenaConfig) -> CliffSensors {
    let radius = ROBOT_RADIUS_M;
    CliffSensors {
        left: cliff_sensor_at(body, 0.0, radius * 0.75, arena),
        front_left: cliff_sensor_at(body, radius * 0.85, radius * 0.45, arena),
        front_right: cliff_sensor_at(body, radius * 0.85, -radius * 0.45, arena),
        right: cliff_sensor_at(body, 0.0, -radius * 0.75, arena),
    }
}

fn cliff_sensor_at(body: &BodySense, local_x: f32, local_y: f32, arena: ArenaConfig) -> f32 {
    let heading = body.odometry.heading_rad;
    let x = body.odometry.x_m + local_x * heading.cos() - local_y * heading.sin();
    let y = body.odometry.y_m + local_x * heading.sin() + local_y * heading.cos();
    let edge_distance = x.min(arena.width_m - x).min(y).min(arena.height_m - y);
    if edge_distance < 0.0 {
        1.0
    } else if edge_distance < 0.04 {
        0.75
    } else if edge_distance < 0.08 {
        0.4
    } else {
        0.0
    }
}

fn charger_contact_rate(body: &BodySense, objects: &[SimObject]) -> f32 {
    objects
        .iter()
        .filter(|object| matches!(object.kind, SimObjectKind::Charger))
        .filter(|object| distance_to_object(body, object) <= ROBOT_RADIUS_M + object.radius_m)
        .map(|object| object.charge_rate.max(0.0))
        .fold(0.0, f32::max)
}

fn charger_near_score(body: &BodySense, objects: &[SimObject]) -> f32 {
    objects
        .iter()
        .filter(|object| matches!(object.kind, SimObjectKind::Charger))
        .map(|object| {
            let distance =
                (distance_to_object(body, object) - object.radius_m - ROBOT_RADIUS_M).max(0.0);
            (1.0 - distance / 1.0).clamp(0.0, 1.0)
        })
        .fold(0.0, f32::max)
}

fn charger_visible_score(body: &BodySense, objects: &[SimObject]) -> f32 {
    let heading = body.odometry.heading_rad;
    objects
        .iter()
        .filter(|object| matches!(object.kind, SimObjectKind::Charger))
        .map(|object| {
            let dx = object.x_m - body.odometry.x_m;
            let dy = object.y_m - body.odometry.y_m;
            let distance = (dx * dx + dy * dy).sqrt();
            if distance > VISIBLE_MAX_M {
                return 0.0;
            }
            let angle = (dy.atan2(dx) - heading + std::f32::consts::PI)
                .rem_euclid(std::f32::consts::TAU)
                - std::f32::consts::PI;
            if angle.abs() > VISIBLE_FOV_RAD * 0.5 {
                return 0.0;
            }
            (1.0 - distance / VISIBLE_MAX_M).clamp(0.0, 1.0)
        })
        .fold(0.0, f32::max)
}

fn distance_to_object(body: &BodySense, object: &SimObject) -> f32 {
    let dx = object.x_m - body.odometry.x_m;
    let dy = object.y_m - body.odometry.y_m;
    (dx * dx + dy * dy).sqrt()
}

fn raycast_distance(
    origin_x: f32,
    origin_y: f32,
    dir_x: f32,
    dir_y: f32,
    objects: &[SimObject],
    arena: ArenaConfig,
    max_range_m: f32,
) -> f32 {
    let mut nearest =
        ray_wall_distance(origin_x, origin_y, dir_x, dir_y, arena).unwrap_or(max_range_m);
    for object in objects.iter().filter(|object| blocks_motion(object)) {
        if let Some(distance) = ray_circle_distance(
            origin_x,
            origin_y,
            dir_x,
            dir_y,
            object.x_m,
            object.y_m,
            object.radius_m + ROBOT_RADIUS_M,
        ) {
            nearest = nearest.min(distance);
        }
    }
    nearest.clamp(0.0, max_range_m)
}

fn ray_wall_distance(
    origin_x: f32,
    origin_y: f32,
    dir_x: f32,
    dir_y: f32,
    arena: ArenaConfig,
) -> Option<f32> {
    let mut hits = Vec::with_capacity(4);
    if dir_x.abs() > f32::EPSILON {
        hits.push((0.0 - origin_x) / dir_x);
        hits.push((arena.width_m - origin_x) / dir_x);
    }
    if dir_y.abs() > f32::EPSILON {
        hits.push((0.0 - origin_y) / dir_y);
        hits.push((arena.height_m - origin_y) / dir_y);
    }
    hits.into_iter()
        .filter(|distance| *distance >= 0.0)
        .filter(|distance| {
            let x = origin_x + dir_x * *distance;
            let y = origin_y + dir_y * *distance;
            x >= -0.001 && x <= arena.width_m + 0.001 && y >= -0.001 && y <= arena.height_m + 0.001
        })
        .min_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal))
}

fn ray_circle_distance(
    origin_x: f32,
    origin_y: f32,
    dir_x: f32,
    dir_y: f32,
    center_x: f32,
    center_y: f32,
    radius: f32,
) -> Option<f32> {
    let oc_x = origin_x - center_x;
    let oc_y = origin_y - center_y;
    let b = 2.0 * (oc_x * dir_x + oc_y * dir_y);
    let c = oc_x * oc_x + oc_y * oc_y - radius * radius;
    let discriminant = b * b - 4.0 * c;
    if discriminant < 0.0 {
        return None;
    }
    let sqrt = discriminant.sqrt();
    let t1 = (-b - sqrt) * 0.5;
    let t2 = (-b + sqrt) * 0.5;
    [t1, t2]
        .into_iter()
        .filter(|distance| *distance >= 0.0)
        .min_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal))
}

fn visible_objects<'a>(
    body: &BodySense,
    objects: &'a [SimObject],
) -> Vec<(&'a SimObject, f32, f32)> {
    objects
        .iter()
        .filter_map(|object| {
            let dx = object.x_m - body.odometry.x_m;
            let dy = object.y_m - body.odometry.y_m;
            let distance = (dx * dx + dy * dy).sqrt();
            if distance > VISIBLE_MAX_M || distance <= f32::EPSILON {
                return None;
            }
            let angle = normalize_angle(dy.atan2(dx) - body.odometry.heading_rad);
            if angle.abs() <= VISIBLE_FOV_RAD * 0.5 {
                Some((object, distance, angle))
            } else {
                None
            }
        })
        .collect()
}

fn project_face_sense(body: &BodySense, objects: &[SimObject]) -> FaceSense {
    let embeddings = visible_objects(body, objects)
        .into_iter()
        .filter_map(|(object, distance, angle)| {
            if let SimObjectKind::Person { identity } = &object.kind {
                Some(sim_embedding(
                    identity.as_deref().unwrap_or(&object.label),
                    distance,
                    angle,
                ))
            } else {
                None
            }
        })
        .collect();
    FaceSense {
        schema_version: 1,
        embeddings,
        vectors: Vec::new(),
    }
}

fn project_voice_sense(body: &BodySense, objects: &[SimObject]) -> VoiceSense {
    let embeddings = audible_objects(body, objects)
        .into_iter()
        .map(|(object, distance)| sim_embedding(&object.label, distance, 0.0))
        .collect();
    VoiceSense {
        schema_version: 1,
        embeddings,
        vectors: Vec::new(),
    }
}

fn project_ear_sense(body: &BodySense, objects: &[SimObject]) -> (EarSense, Option<PcmAudioFrame>) {
    let audible = audible_objects(body, objects);
    let transcript = audible
        .first()
        .map(|(object, _)| format!("{} sound", object.label));
    let features = audible
        .iter()
        .map(|(object, distance)| {
            vec![
                1.0 / (1.0 + *distance),
                label_hash(&object.label),
                object.x_m,
                object.y_m,
            ]
        })
        .collect::<Vec<_>>();
    let pcm = if audible.is_empty() {
        None
    } else {
        let samples = (0..256)
            .map(|index| {
                let wave = ((index as f32 / 8.0).sin() * 10_000.0) as i16;
                wave
            })
            .collect();
        Some(PcmAudioFrame {
            captured_at_ms: body.last_update_ms,
            sample_rate_hz: 16_000,
            channels: 1,
            samples,
        })
    };
    (
        EarSense {
            schema_version: 1,
            features,
            transcript,
            ..EarSense::default()
        },
        pcm,
    )
}

fn audible_objects<'a>(body: &BodySense, objects: &'a [SimObject]) -> Vec<(&'a SimObject, f32)> {
    objects
        .iter()
        .filter(|object| {
            object.emits_sound
                || matches!(
                    object.kind,
                    SimObjectKind::SoundSource { .. } | SimObjectKind::Person { .. }
                )
        })
        .filter_map(|object| {
            let distance = distance_to_object(body, object);
            (distance <= 3.0).then_some((object, distance))
        })
        .collect()
}

fn project_kinect_sense(
    body: &BodySense,
    objects: &[SimObject],
    arena: ArenaConfig,
) -> KinectSense {
    let depth_m = project_range_beams(body, objects, arena, 32, RANGE_FOV_RAD, RANGE_MAX_M)
        .into_iter()
        .map(|normalized| normalized * RANGE_MAX_M)
        .collect::<Vec<_>>();
    let color_features = visible_objects(body, objects)
        .into_iter()
        .map(|(object, distance, angle)| {
            vec![
                object.color_rgb[0] as f32 / 255.0,
                object.color_rgb[1] as f32 / 255.0,
                object.color_rgb[2] as f32 / 255.0,
                distance,
                angle,
            ]
        })
        .collect();
    let skeletons = visible_objects(body, objects)
        .into_iter()
        .filter_map(|(object, distance, angle)| match object.kind {
            SimObjectKind::Person { .. } => Some(KinectSkeletonSense {
                tracking_id: label_tracking_id(&object.id),
                lean_xy: [angle.sin(), 0.0],
                joints: vec![KinectJointSense {
                    joint_name: "center".to_string(),
                    position_m: [distance, angle.sin() * distance, 1.0],
                    tracking_confidence: 0.8,
                    tracked: true,
                }],
            }),
            _ => None,
        })
        .collect();
    KinectSense {
        schema_version: 1,
        color_features,
        depth_m,
        skeletons,
        ..KinectSense::default()
    }
}

fn normalize_angle(mut angle: f32) -> f32 {
    while angle > std::f32::consts::PI {
        angle -= std::f32::consts::TAU;
    }
    while angle < -std::f32::consts::PI {
        angle += std::f32::consts::TAU;
    }
    angle
}

fn sim_embedding(label: &str, distance: f32, angle: f32) -> Vec<f32> {
    vec![
        label_hash(label),
        (1.0 / (1.0 + distance)).clamp(0.0, 1.0),
        angle.sin(),
        angle.cos(),
    ]
}

fn label_hash(label: &str) -> f32 {
    let hash = label.bytes().fold(0u32, |acc, byte| {
        acc.wrapping_mul(31).wrapping_add(byte as u32)
    });
    (hash % 1_000) as f32 / 1_000.0
}

fn label_tracking_id(label: &str) -> u64 {
    label.bytes().fold(17u64, |acc, byte| {
        acc.wrapping_mul(37).wrapping_add(byte as u64)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arena() -> ArenaConfig {
        ArenaConfig {
            width_m: 4.0,
            height_m: 4.0,
        }
    }

    fn centered_body() -> BodySense {
        let mut body = BodySense::default();
        body.odometry.x_m = 2.0;
        body.odometry.y_m = 2.0;
        body.odometry.heading_rad = 0.0;
        body.last_update_ms = 42;
        body
    }

    fn changed_pixels(left: &EyeFrame, right: &EyeFrame) -> usize {
        left.bytes
            .chunks_exact(3)
            .zip(right.bytes.chunks_exact(3))
            .filter(|(a, b)| a != b)
            .count()
    }

    fn color_like_pixels(frame: &EyeFrame, color: [u8; 3]) -> usize {
        frame
            .bytes
            .chunks_exact(3)
            .filter(|pixel| {
                pixel[0].abs_diff(color[0]) < 45
                    && pixel[1].abs_diff(color[1]) < 45
                    && pixel[2].abs_diff(color[2]) < 45
            })
            .count()
    }

    fn colored_bbox(frame: &EyeFrame, color: [u8; 3]) -> Option<(usize, usize, usize, usize)> {
        let mut min_x = frame.width as usize;
        let mut min_y = frame.height as usize;
        let mut max_x = 0usize;
        let mut max_y = 0usize;
        let mut found = false;
        for (index, pixel) in frame.bytes.chunks_exact(3).enumerate() {
            if pixel[0].abs_diff(color[0]) < 45
                && pixel[1].abs_diff(color[1]) < 45
                && pixel[2].abs_diff(color[2]) < 45
            {
                let x = index % frame.width as usize;
                let y = index / frame.width as usize;
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
                found = true;
            }
        }
        found.then_some((min_x, min_y, max_x, max_y))
    }

    #[tokio::test]
    async fn robot_cannot_pass_through_walls_and_sets_bump() {
        let (_world, mut motor) = VirtualWorld::new_with_motor(0, arena());
        {
            let mut guard = motor.state.lock().unwrap();
            guard.snapshot.body.odometry.x_m = 0.2;
            guard.snapshot.body.odometry.y_m = 2.0;
            guard.snapshot.body.odometry.heading_rad = std::f32::consts::PI;
        }

        let body = motor
            .send(MotionCommand::Forward { speed_m_s: 1.0 })
            .await
            .unwrap();

        assert!(body.odometry.x_m >= 0.2 - f32::EPSILON);
        assert!(body.flags.bump_left && body.flags.bump_right);
        assert!(body.flags.wall);
    }

    #[tokio::test]
    async fn robot_cannot_pass_through_obstacle_discs() {
        let (mut world, mut motor) = VirtualWorld::new_with_motor(0, arena());
        world.add_object(SimObject::obstacle("box", "box", 1.0, 0.0, 0.25));
        {
            let mut guard = motor.state.lock().unwrap();
            guard.snapshot.body.odometry.x_m = 0.6;
            guard.snapshot.body.odometry.y_m = 0.0;
        }

        let body = motor
            .send(MotionCommand::Forward { speed_m_s: 2.0 })
            .await
            .unwrap();

        assert!(body.odometry.x_m < 0.8);
        assert!(body.flags.bump_left || body.flags.bump_right);
    }

    #[tokio::test]
    async fn charger_contact_increases_battery() {
        let (mut world, mut motor) = VirtualWorld::new_with_motor(0, arena());
        world.add_object(SimObject {
            charge_rate: 0.5,
            ..SimObject::charger("dock", "dock", 2.0, 2.0, 0.5)
        });
        {
            let mut guard = motor.state.lock().unwrap();
            guard.snapshot.body.battery_level = 0.4;
        }

        let body = motor.send(MotionCommand::Stop).await.unwrap();

        assert!(body.charging);
        assert!(body.battery_level > 0.4);
    }

    #[tokio::test]
    async fn edge_projects_front_cliff_sensors() {
        let (mut world, _motor) = VirtualWorld::new_with_motor(0, arena());
        {
            let mut guard = world.state.lock().unwrap();
            guard.snapshot.body.odometry.x_m = 3.95;
            guard.snapshot.body.odometry.y_m = 2.0;
            guard.snapshot.body.odometry.heading_rad = 0.0;
        }

        let snapshot = world.snapshot().await.unwrap();

        assert!(snapshot.body.cliff_sensors.front_left >= 0.5);
        assert!(snapshot.body.cliff_sensors.front_right >= 0.5);
        assert!(snapshot.body.flags.cliff_front_left);
        assert!(snapshot.body.flags.cliff_front_right);
    }

    #[test]
    fn ray_beams_differ_by_direction() {
        let mut body = BodySense::default();
        body.odometry.x_m = 2.0;
        body.odometry.y_m = 2.0;
        let objects = vec![SimObject::obstacle("front", "front", 2.8, 2.0, 0.2)];

        let beams = project_range_beams(&body, &objects, arena(), 5, std::f32::consts::PI, 4.0);

        assert!(beams[2] < beams[0]);
        assert!(beams[2] < beams[4]);
    }

    #[test]
    fn eye_frame_has_camera_dimensions_and_rgb8_format() {
        let frame = project_eye_frame(&centered_body(), &[], arena());

        assert_eq!(frame.width, EYE_WIDTH as u32);
        assert_eq!(frame.height, EYE_HEIGHT as u32);
        assert_eq!(frame.format, EyeFrameFormat::Rgb8);
        assert_eq!(frame.bytes.len(), EYE_WIDTH * EYE_HEIGHT * 3);
    }

    #[test]
    fn empty_room_eye_shows_floor_and_horizon() {
        let frame = project_eye_frame(&centered_body(), &[], arena());
        let unique = frame
            .bytes
            .chunks_exact(3)
            .map(|pixel| [pixel[0], pixel[1], pixel[2]])
            .collect::<std::collections::BTreeSet<_>>();
        let mean_luma = frame
            .bytes
            .chunks_exact(3)
            .map(|pixel| (pixel[0] as u32 + pixel[1] as u32 + pixel[2] as u32) / 3)
            .sum::<u32>() as f32
            / (frame.width * frame.height) as f32;

        assert!(unique.len() > 20);
        assert!(mean_luma > 35.0);
    }

    #[test]
    fn object_in_front_changes_eye_pixels() {
        let body = centered_body();
        let empty = project_eye_frame(&body, &[], arena());
        let object = SimObject::obstacle("front", "front", 3.0, 2.0, 0.25);
        let with_object = project_eye_frame(&body, &[object], arena());

        assert!(changed_pixels(&empty, &with_object) > 100);
    }

    #[test]
    fn object_behind_robot_does_not_render() {
        let body = centered_body();
        let empty = project_eye_frame(&body, &[], arena());
        let object = SimObject::obstacle("behind", "behind", 1.0, 2.0, 0.25);
        let with_object = project_eye_frame(&body, &[object], arena());

        assert_eq!(changed_pixels(&empty, &with_object), 0);
    }

    #[test]
    fn object_outside_horizontal_fov_does_not_render() {
        let body = centered_body();
        let empty = project_eye_frame(&body, &[], arena());
        let object = SimObject::obstacle("side", "side", 2.6, 3.8, 0.2);
        let with_object = project_eye_frame(&body, &[object], arena());

        assert_eq!(changed_pixels(&empty, &with_object), 0);
    }

    #[test]
    fn nearer_object_appears_larger_than_farther_object() {
        let body = centered_body();
        let near = SimObject::obstacle("near", "near", 2.9, 2.0, 0.25);
        let far = SimObject::obstacle("far", "far", 3.7, 2.0, 0.25);
        let near_frame = project_eye_frame(&body, &[near], arena());
        let far_frame = project_eye_frame(&body, &[far], arena());
        let near_bbox = colored_bbox(&near_frame, [180, 90, 80]).unwrap();
        let far_bbox = colored_bbox(&far_frame, [180, 90, 80]).unwrap();
        let near_area = (near_bbox.2 - near_bbox.0 + 1) * (near_bbox.3 - near_bbox.1 + 1);
        let far_area = (far_bbox.2 - far_bbox.0 + 1) * (far_bbox.3 - far_bbox.1 + 1);

        assert!(near_area > far_area);
    }

    #[test]
    fn charger_person_and_obstacle_colors_render_in_front() {
        let body = centered_body();
        let objects = vec![
            SimObject::charger("dock", "dock", 3.0, 1.4, 0.25),
            SimObject {
                id: "person".to_string(),
                label: "person".to_string(),
                kind: SimObjectKind::Person { identity: None },
                x_m: 3.0,
                y_m: 2.0,
                radius_m: 0.2,
                color_rgb: [220, 180, 140],
                emits_sound: false,
                charge_rate: 0.0,
            },
            SimObject::obstacle("box", "box", 3.0, 2.6, 0.25),
        ];
        let frame = project_eye_frame(&body, &objects, arena());

        assert!(color_like_pixels(&frame, [80, 220, 130]) > 8);
        assert!(color_like_pixels(&frame, [220, 180, 140]) > 20);
        assert!(color_like_pixels(&frame, [180, 90, 80]) > 20);
    }

    #[test]
    fn eye_projection_is_deterministic() {
        let body = centered_body();
        let objects = vec![SimObject::obstacle("front", "front", 3.0, 2.0, 0.25)];

        let left = project_eye_frame(&body, &objects, arena());
        let right = project_eye_frame(&body, &objects, arena());

        assert_eq!(left, right);
    }

    #[tokio::test]
    async fn visible_person_projects_face_and_kinect_skeleton() {
        let (mut world, _motor) = VirtualWorld::new_with_motor(0, arena());
        world.add_object(SimObject {
            id: "jes".to_string(),
            label: "Jes".to_string(),
            kind: SimObjectKind::Person {
                identity: Some("Jes".to_string()),
            },
            x_m: 3.0,
            y_m: 2.0,
            radius_m: 0.2,
            color_rgb: [220, 180, 140],
            emits_sound: false,
            charge_rate: 0.0,
        });

        let snapshot = world.snapshot().await.unwrap();

        assert_eq!(snapshot.face.embeddings.len(), 1);
        assert_eq!(snapshot.kinect.skeletons.len(), 1);
    }

    #[tokio::test]
    async fn sound_source_projects_ear_and_voice() {
        let (mut world, _motor) = VirtualWorld::new_with_motor(0, arena());
        world.add_object(SimObject {
            id: "speaker".to_string(),
            label: "speaker".to_string(),
            kind: SimObjectKind::SoundSource {
                label: "speaker".to_string(),
            },
            x_m: 1.0,
            y_m: 0.0,
            radius_m: 0.1,
            color_rgb: [80, 80, 220],
            emits_sound: true,
            charge_rate: 0.0,
        });

        let snapshot = world.snapshot().await.unwrap();

        assert!(!snapshot.ear.features.is_empty());
        assert!(snapshot
            .ear
            .transcript
            .as_deref()
            .unwrap_or_default()
            .contains("speaker"));
        assert_eq!(snapshot.voice.embeddings.len(), 1);
    }
}
