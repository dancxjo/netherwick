use anyhow::Result;
use async_trait::async_trait;
use netherwick_body::{
    BodySense, CliffSensors, MotionCommand, MotorCommand, MotorComplex, RobotBody,
};
use netherwick_now::{
    EarSense, ExtensionSense, FaceSense, GpsSense, ImuSense, KinectJointSense, KinectSense,
    KinectSkeletonSense, ObjectClass, ObjectObservation, ObjectObservationSource, ObjectSense,
    RangeSense, VoiceSense,
};
use netherwick_sensors::{
    EyeFrame, EyeFrameFormat, PcmAudioFrame, World, WorldSnapshot, WorldUpdate,
};
use rand::{rngs::StdRng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

pub mod scenario;
pub use scenario::{
    build_scenario, default_sim_world, mutate_dream_world, DreamConfig, ScenarioConfig,
    ScenarioKind, ScenarioMetadata, ScenarioWorld,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spoken_text: Option<String>,
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
            spoken_text: None,
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
            spoken_text: None,
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
    last_motion_sent: Option<MotionCommand>,
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
            last_motion_sent: None,
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

    pub fn last_motion_sent(&self) -> Option<MotionCommand> {
        self.state
            .lock()
            .expect("virtual world mutex poisoned")
            .last_motion_sent
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
            let fallback = project_blank_eye_frame(body);
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
        state.snapshot.objects = project_object_sense(body, &state.objects);
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
        guard.last_motion_sent = Some(command.clone());
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

fn project_blank_eye_frame(body: &BodySense) -> EyeFrame {
    EyeFrame {
        source: Some("rust-sim-blank".to_string()),
        captured_at_ms: body.last_update_ms,
        width: EYE_WIDTH as u32,
        height: EYE_HEIGHT as u32,
        format: EyeFrameFormat::Rgb8,
        bytes: vec![128; EYE_WIDTH * EYE_HEIGHT * 3],
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

fn project_object_sense(body: &BodySense, objects: &[SimObject]) -> ObjectSense {
    let observations = visible_objects(body, objects)
        .into_iter()
        .map(|(object, distance, bearing)| ObjectObservation {
            label: object.label.clone(),
            class: sim_object_class(&object.kind),
            bearing_rad: bearing,
            distance_m: Some((distance - object.radius_m - ROBOT_RADIUS_M).max(0.0)),
            confidence: visible_object_confidence(distance, bearing),
            source: ObjectObservationSource::Sim,
        })
        .collect();
    ObjectSense {
        schema_version: 1,
        observations,
    }
}

fn sim_object_class(kind: &SimObjectKind) -> ObjectClass {
    match kind {
        SimObjectKind::Obstacle => ObjectClass::Obstacle,
        SimObjectKind::Charger => ObjectClass::Charger,
        SimObjectKind::Person { .. } => ObjectClass::Person,
        SimObjectKind::SoundSource { .. } => ObjectClass::SoundSource,
        SimObjectKind::Landmark { .. } => ObjectClass::Landmark,
    }
}

fn visible_object_confidence(distance_m: f32, bearing_rad: f32) -> f32 {
    let distance_score = (1.0 - distance_m / VISIBLE_MAX_M).clamp(0.0, 1.0);
    let bearing_score = (1.0 - bearing_rad.abs() / (VISIBLE_FOV_RAD * 0.5)).clamp(0.0, 1.0);
    (0.4 + 0.4 * distance_score + 0.2 * bearing_score).clamp(0.0, 1.0)
}

fn project_ear_sense(body: &BodySense, objects: &[SimObject]) -> (EarSense, Option<PcmAudioFrame>) {
    let audible = audible_objects(body, objects);
    let transcript = audible
        .first()
        .map(|(object, _)| dream_speech_transcript(object));
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
        let phrase = audible
            .first()
            .map(|(object, _)| dream_speech_text(object))
            .unwrap_or_default();
        let seed = phrase.bytes().fold(0u32, |acc, byte| {
            acc.wrapping_mul(31).wrapping_add(byte as u32)
        });
        let frequency = 5.0 + (seed % 17) as f32;
        let wobble = 1.0 + ((seed / 17) % 5) as f32 * 0.07;
        let samples = (0..256)
            .map(|index| {
                let envelope = 1.0 - index as f32 / 256.0;
                let wave = (((index as f32 / frequency).sin()
                    * (index as f32 / 19.0 * wobble).cos())
                    * 10_000.0
                    * envelope) as i16;
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

fn dream_speech_transcript(object: &SimObject) -> String {
    let speech = dream_speech_text(object);
    if matches!(object.kind, SimObjectKind::Person { .. }) {
        format!("{} says \"{speech}\"", object.label)
    } else {
        format!("{} whispers \"{speech}\"", object.label)
    }
}

fn dream_speech_text(object: &SimObject) -> String {
    object
        .spoken_text
        .clone()
        .unwrap_or_else(|| match &object.kind {
            SimObjectKind::Person { .. } => "I saw you in the dream world.".to_string(),
            SimObjectKind::SoundSource { label } => format!("{label} is calling from far away."),
            _ => format!("{} is making a sound.", object.label),
        })
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
        let frame = project_blank_eye_frame(&centered_body());

        assert_eq!(frame.width, EYE_WIDTH as u32);
        assert_eq!(frame.height, EYE_HEIGHT as u32);
        assert_eq!(frame.format, EyeFrameFormat::Rgb8);
        assert_eq!(frame.bytes.len(), EYE_WIDTH * EYE_HEIGHT * 3);
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
            spoken_text: None,
            charge_rate: 0.0,
        });

        let snapshot = world.snapshot().await.unwrap();

        assert_eq!(snapshot.face.embeddings.len(), 1);
        assert_eq!(snapshot.kinect.skeletons.len(), 1);
    }

    #[tokio::test]
    async fn visible_object_projects_typed_observation_into_now() {
        let (mut world, _motor) = VirtualWorld::new_with_motor(0, arena());
        world.add_object(SimObject::charger("dock", "charger dock", 3.0, 2.0, 0.2));

        let snapshot = world.snapshot().await.unwrap();
        let observation = snapshot
            .objects
            .observations
            .iter()
            .find(|observation| observation.label == "charger dock")
            .unwrap();

        assert_eq!(observation.class, ObjectClass::Charger);
        assert_eq!(observation.source, ObjectObservationSource::Sim);
        assert!(observation.bearing_rad.abs() < 0.01);
        assert!(observation.distance_m.unwrap() > 0.5);
        assert!(observation.confidence > 0.5);

        let now = snapshot.to_now(snapshot.body.last_update_ms);
        assert_eq!(now.objects.observations, snapshot.objects.observations);
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
            spoken_text: Some("the door is dreaming".to_string()),
            charge_rate: 0.0,
        });

        let snapshot = world.snapshot().await.unwrap();

        assert!(!snapshot.ear.features.is_empty());
        assert!(snapshot
            .ear
            .transcript
            .as_deref()
            .unwrap_or_default()
            .contains("the door is dreaming"));
        assert_eq!(snapshot.voice.embeddings.len(), 1);
    }
}
