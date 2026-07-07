use pete_body::BodySense;
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::{ArenaConfig, SimCockpit, SimObject, SimObjectKind, VirtualWorld};

pub const ROBOT_SPAWN_CLEARANCE_M: f32 = 0.45;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScenarioKind {
    EmptyRoom,
    ObstacleAvoidance,
    CornerTrap,
    ColumnTrap,
    ChargerSeeking,
    PersonAndSpeaker,
    MixedRoom,
    Dream,
}

impl ScenarioKind {
    pub fn slug(self) -> &'static str {
        match self {
            Self::EmptyRoom => "empty-room",
            Self::ObstacleAvoidance => "obstacle-avoidance",
            Self::CornerTrap => "corner-trap",
            Self::ColumnTrap => "column-trap",
            Self::ChargerSeeking => "charger-seeking",
            Self::PersonAndSpeaker => "person-speaker-room",
            Self::MixedRoom => "mixed-room",
            Self::Dream => "dream",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DreamConfig {
    pub seed: u64,
    pub weirdness: f32,
    pub density: f32,
    pub sociality: f32,
    pub hazard_bias: f32,
    pub charger_bias: f32,
}

impl DreamConfig {
    pub fn from_env(seed: u64) -> Self {
        Self {
            seed,
            weirdness: env_f32("PETE_DREAM_WEIRDNESS", 0.35),
            density: env_f32("PETE_DREAM_DENSITY", 0.5),
            sociality: env_f32("PETE_DREAM_SOCIALITY", 0.3),
            hazard_bias: env_f32("PETE_DREAM_HAZARD_BIAS", 0.25),
            charger_bias: env_f32("PETE_DREAM_CHARGER_BIAS", 0.2),
        }
        .clamped()
    }

    pub fn clamped(self) -> Self {
        Self {
            seed: self.seed,
            weirdness: self.weirdness.clamp(0.0, 1.0),
            density: self.density.clamp(0.0, 1.0),
            sociality: self.sociality.clamp(0.0, 1.0),
            hazard_bias: self.hazard_bias.clamp(0.0, 1.0),
            charger_bias: self.charger_bias.clamp(0.0, 1.0),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScenarioConfig {
    pub kind: ScenarioKind,
    pub seed: u64,
    pub arena: ArenaConfig,
    pub object_count: usize,
    pub charger_count: usize,
    pub obstacle_count: usize,
    pub person_count: usize,
    pub speaker_count: usize,
    pub landmark_count: usize,
    pub dream: Option<DreamConfig>,
}

impl ScenarioConfig {
    pub fn new(kind: ScenarioKind, seed: u64) -> Self {
        let mut config = Self {
            kind,
            seed,
            arena: ArenaConfig {
                width_m: 8.0,
                height_m: 8.0,
            },
            object_count: 0,
            charger_count: 0,
            obstacle_count: 0,
            person_count: 0,
            speaker_count: 0,
            landmark_count: 0,
            dream: None,
        };
        match kind {
            ScenarioKind::EmptyRoom => {
                config.charger_count = 1;
            }
            ScenarioKind::ObstacleAvoidance => {
                config.charger_count = 1;
                config.obstacle_count = 7;
            }
            ScenarioKind::CornerTrap => {
                config.charger_count = 1;
                config.obstacle_count = 3;
            }
            ScenarioKind::ColumnTrap => {
                config.charger_count = 1;
                config.obstacle_count = 1;
            }
            ScenarioKind::ChargerSeeking => {
                config.charger_count = 2;
                config.obstacle_count = 4;
            }
            ScenarioKind::PersonAndSpeaker => {
                config.charger_count = 1;
                config.person_count = 1;
                config.obstacle_count = 2;
            }
            ScenarioKind::MixedRoom => {
                config.charger_count = 1;
                config.obstacle_count = 5;
                config.person_count = 1;
            }
            ScenarioKind::Dream => {
                let dream = DreamConfig::from_env(seed);
                config.dream = Some(dream);
                apply_dream_config(&mut config, dream);
            }
        }
        config.object_count = config.charger_count
            + config.obstacle_count
            + config.person_count
            + config.speaker_count
            + config.landmark_count;
        config
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScenarioMetadata {
    pub kind: ScenarioKind,
    pub seed: u64,
    pub arena: ArenaConfig,
    pub body: BodySense,
    pub objects: Vec<SimObject>,
}

#[derive(Clone, Debug)]
pub struct ScenarioWorld {
    pub world: VirtualWorld,
    pub motors: SimCockpit,
    pub metadata: ScenarioMetadata,
}

pub fn default_sim_world(seed: u64) -> (VirtualWorld, SimCockpit) {
    let scenario = build_scenario(ScenarioConfig::new(ScenarioKind::Dream, seed));
    (scenario.world, scenario.motors)
}

pub fn build_scenario(config: ScenarioConfig) -> ScenarioWorld {
    let mut rng = StdRng::seed_from_u64(config.seed);
    let (mut world, motors) = VirtualWorld::new_with_cockpit(config.seed, config.arena);
    let body = spawn_body(&config, &mut rng);
    world.set_body(body.clone());

    let mut objects = Vec::new();
    if config.kind == ScenarioKind::Dream {
        add_dream_objects(&config, &mut rng, &body, &mut objects);
    } else {
        add_kind_objects(&config, &mut rng, &body, &mut objects);
    }
    world.set_objects(objects.clone());

    ScenarioWorld {
        world,
        motors,
        metadata: ScenarioMetadata {
            kind: config.kind,
            seed: config.seed,
            arena: config.arena,
            body,
            objects,
        },
    }
}

fn spawn_body(config: &ScenarioConfig, rng: &mut StdRng) -> BodySense {
    let mut body = BodySense::default();
    body.last_update_ms = config.seed;
    body.odometry.heading_rad = rng.gen_range(-std::f32::consts::PI..std::f32::consts::PI);
    match config.kind {
        ScenarioKind::ObstacleAvoidance => {
            body.odometry.x_m = 1.0;
            body.odometry.y_m = config.arena.height_m * 0.5;
            body.odometry.heading_rad = 0.0;
        }
        ScenarioKind::CornerTrap => {
            body.odometry.x_m = 0.42;
            body.odometry.y_m = 0.42;
            body.odometry.heading_rad = -2.35;
        }
        ScenarioKind::ColumnTrap => {
            body.odometry.x_m = 1.0;
            body.odometry.y_m = config.arena.height_m * 0.5;
            body.odometry.heading_rad = 0.0;
        }
        ScenarioKind::ChargerSeeking => {
            body.battery_level = 0.18;
            body.odometry.x_m = rng.gen_range(0.8..config.arena.width_m * 0.35);
            body.odometry.y_m = rng.gen_range(0.8..config.arena.height_m - 0.8);
            body.odometry.heading_rad = rng.gen_range(-0.7..0.7);
        }
        ScenarioKind::PersonAndSpeaker => {
            body.odometry.x_m = 1.2;
            body.odometry.y_m = config.arena.height_m * 0.5;
            body.odometry.heading_rad = 0.0;
        }
        ScenarioKind::EmptyRoom | ScenarioKind::MixedRoom => {
            body.odometry.x_m = rng.gen_range(0.8..config.arena.width_m - 0.8);
            body.odometry.y_m = rng.gen_range(0.8..config.arena.height_m - 0.8);
        }
        ScenarioKind::Dream => {
            body.battery_level = rng.gen_range(0.16..0.88);
            body.odometry.x_m = rng.gen_range(0.9..config.arena.width_m - 0.9);
            body.odometry.y_m = rng.gen_range(0.9..config.arena.height_m - 0.9);
        }
    }
    body
}

fn add_kind_objects(
    config: &ScenarioConfig,
    rng: &mut StdRng,
    body: &BodySense,
    objects: &mut Vec<SimObject>,
) {
    for index in 0..config.obstacle_count {
        let (x_m, y_m) = if config.kind == ScenarioKind::ObstacleAvoidance && index == 0 {
            (body.odometry.x_m + 0.55, body.odometry.y_m)
        } else if config.kind == ScenarioKind::ColumnTrap {
            (body.odometry.x_m + 0.50, body.odometry.y_m)
        } else if config.kind == ScenarioKind::CornerTrap {
            match index {
                0 => (0.42, 1.02),
                1 => (1.02, 0.42),
                _ => (0.92, 0.92),
            }
        } else {
            random_free_position(config.arena, rng, body, objects, 0.35)
        };
        objects.push(SimObject::obstacle(
            format!("obstacle-{index}"),
            format!("obstacle {index}"),
            x_m,
            y_m,
            if config.kind == ScenarioKind::ColumnTrap {
                0.28
            } else {
                rng.gen_range(0.22..0.42)
            },
        ));
    }

    for index in 0..config.charger_count {
        let (x_m, y_m) = if config.kind == ScenarioKind::ChargerSeeking && index == 0 {
            (
                rng.gen_range(config.arena.width_m * 0.55..config.arena.width_m - 0.8),
                rng.gen_range(0.8..config.arena.height_m - 0.8),
            )
        } else {
            random_free_position(config.arena, rng, body, objects, 0.25)
        };
        objects.push(SimObject::charger(
            format!("charger-{index}"),
            format!("charger {index}"),
            x_m,
            y_m,
            0.25,
        ));
    }

    for index in 0..config.person_count {
        let (x_m, y_m) = if config.kind == ScenarioKind::PersonAndSpeaker && index == 0 {
            (2.8, body.odometry.y_m)
        } else {
            random_free_position(config.arena, rng, body, objects, 0.24)
        };
        let name = dream_person_name(index);
        objects.push(SimObject {
            id: format!("person-{index}"),
            label: name.to_string(),
            kind: SimObjectKind::Person {
                identity: Some(name.to_string()),
            },
            x_m,
            y_m,
            radius_m: 0.22,
            color_rgb: [220, 180, 140],
            emits_sound: true,
            spoken_text: Some(dream_person_phrase(index).to_string()),
            charge_rate: 0.0,
        });
    }

    for index in 0..config.speaker_count {
        let (x_m, y_m) = if config.kind == ScenarioKind::PersonAndSpeaker && index == 0 {
            (2.4, body.odometry.y_m + 0.7)
        } else {
            random_free_position(config.arena, rng, body, objects, 0.16)
        };
        objects.push(SimObject {
            id: format!("speaker-{index}"),
            label: format!("speaker {index}"),
            kind: SimObjectKind::SoundSource {
                label: format!("speaker {index}"),
            },
            x_m,
            y_m,
            radius_m: 0.12,
            color_rgb: [80, 80, 220],
            emits_sound: true,
            spoken_text: Some(dream_voice_phrase(index).to_string()),
            charge_rate: 0.0,
        });
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DreamLayout {
    Room,
    Corridor,
    SmallRoom,
    ColumnField,
    Islands,
    Asymmetric,
}

fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(default)
}

fn apply_dream_config(config: &mut ScenarioConfig, dream: DreamConfig) {
    let mut rng = StdRng::seed_from_u64(dream.seed ^ 0xD3EA_AA11_5EED);
    let layout = choose_dream_layout(&mut rng, dream);
    config.arena = match layout {
        DreamLayout::Corridor => ArenaConfig {
            width_m: rng.gen_range(9.0..14.0),
            height_m: rng.gen_range(5.0..7.0),
        },
        DreamLayout::SmallRoom => ArenaConfig {
            width_m: rng.gen_range(5.0..7.0),
            height_m: rng.gen_range(5.0..7.0),
        },
        DreamLayout::ColumnField => ArenaConfig {
            width_m: rng.gen_range(7.0..11.0),
            height_m: rng.gen_range(7.0..11.0),
        },
        DreamLayout::Islands | DreamLayout::Asymmetric | DreamLayout::Room => ArenaConfig {
            width_m: rng.gen_range(7.0..14.0),
            height_m: rng.gen_range(7.0..14.0),
        },
    };

    let area_factor = (config.arena.width_m * config.arena.height_m / 64.0).clamp(0.7, 2.2);
    let density = dream.density;
    let weirdness = dream.weirdness;
    config.obstacle_count =
        ((4.0 + density * 9.0 + dream.hazard_bias * 4.0) * area_factor) as usize;
    config.charger_count = if rng.gen_bool((0.35 + dream.charger_bias as f64 * 0.65).min(0.95)) {
        1 + (density > 0.72 && rng.gen_bool(0.35)) as usize
    } else {
        0
    };
    config.person_count =
        (dream.sociality * 3.0 + if weirdness > 0.65 { 1.0 } else { 0.0 }).round() as usize;
    config.speaker_count =
        (weirdness * 3.0 + dream.sociality * 1.5 + density * 0.5).round() as usize;
    config.landmark_count = (weirdness * 4.0 + density * 1.5).round() as usize;

    if density + dream.sociality + dream.charger_bias > 0.55
        && config.charger_count + config.person_count + config.speaker_count + config.landmark_count
            == 0
    {
        config.charger_count = 1;
    }
}

fn choose_dream_layout(rng: &mut StdRng, dream: DreamConfig) -> DreamLayout {
    let roll = rng.gen_range(0.0..1.0);
    let weird = dream.weirdness;
    if roll < 0.15 {
        DreamLayout::Corridor
    } else if roll < 0.30 {
        DreamLayout::SmallRoom
    } else if roll < 0.48 + weird * 0.08 {
        DreamLayout::ColumnField
    } else if roll < 0.66 + weird * 0.12 {
        DreamLayout::Islands
    } else if roll < 0.82 + weird * 0.12 {
        DreamLayout::Asymmetric
    } else {
        DreamLayout::Room
    }
}

fn add_dream_objects(
    config: &ScenarioConfig,
    rng: &mut StdRng,
    body: &BodySense,
    objects: &mut Vec<SimObject>,
) {
    let dream = config
        .dream
        .unwrap_or_else(|| DreamConfig::from_env(config.seed));
    let layout = choose_dream_layout(rng, dream);

    for index in 0..config.obstacle_count {
        let radius_m = rng.gen_range(0.18..0.38 + dream.weirdness * 0.22);
        if let Some((x_m, y_m)) =
            dream_position(config, rng, body, objects, radius_m, layout, index)
        {
            let label = choose(rng, DREAM_OBSTACLE_LABELS);
            let mut object = SimObject::obstacle(
                format!("dream-obstacle-{index}"),
                label.to_string(),
                x_m,
                y_m,
                radius_m,
            );
            object.color_rgb = dream_color(rng, dream, [180, 90, 80]);
            objects.push(object);
        }
    }

    for index in 0..config.charger_count {
        let radius_m = rng.gen_range(0.20..0.32);
        if let Some((x_m, y_m)) =
            dream_position(config, rng, body, objects, radius_m, layout, index)
        {
            let label = choose(rng, DREAM_CHARGER_LABELS);
            let mut charger = SimObject::charger(
                format!("dream-charger-{index}"),
                label.to_string(),
                x_m,
                y_m,
                radius_m,
            );
            charger.color_rgb = dream_color(rng, dream, [80, 220, 130]);
            objects.push(charger);
            if dream.hazard_bias > 0.2 && rng.gen_bool((0.25 + dream.weirdness * 0.35) as f64) {
                add_misleading_obstacle_near(config, rng, body, objects, x_m, y_m, dream);
            }
        }
    }

    for index in 0..config.person_count {
        let radius_m = rng.gen_range(0.20..0.27);
        if let Some((x_m, y_m)) =
            dream_position(config, rng, body, objects, radius_m, layout, index)
        {
            let name = choose(rng, DREAM_PERSON_NAMES).to_string();
            objects.push(SimObject {
                id: format!("dream-person-{index}"),
                label: name.clone(),
                kind: SimObjectKind::Person {
                    identity: Some(name),
                },
                x_m,
                y_m,
                radius_m,
                color_rgb: dream_color(rng, dream, [220, 180, 140]),
                emits_sound: true,
                spoken_text: Some(choose(rng, DREAM_PERSON_PHRASES).to_string()),
                charge_rate: 0.0,
            });
        }
    }

    for index in 0..config.speaker_count {
        let radius_m = rng.gen_range(0.10..0.18);
        if let Some((x_m, y_m)) =
            dream_position(config, rng, body, objects, radius_m, layout, index)
        {
            let label = choose(rng, DREAM_VOICE_LABELS).to_string();
            objects.push(SimObject {
                id: format!("dream-speaker-{index}"),
                label: label.clone(),
                kind: SimObjectKind::SoundSource {
                    label: label.clone(),
                },
                x_m,
                y_m,
                radius_m,
                color_rgb: dream_color(rng, dream, [80, 80, 220]),
                emits_sound: true,
                spoken_text: Some(choose(rng, DREAM_VOICE_PHRASES).to_string()),
                charge_rate: 0.0,
            });
        }
    }

    for index in 0..config.landmark_count {
        let radius_m = rng.gen_range(0.24..0.55 + dream.weirdness * 0.25);
        if let Some((x_m, y_m)) =
            dream_position(config, rng, body, objects, radius_m, layout, index)
        {
            let label = choose(rng, DREAM_LANDMARK_LABELS).to_string();
            objects.push(SimObject {
                id: format!("dream-landmark-{index}"),
                label: label.clone(),
                kind: SimObjectKind::Landmark { label },
                x_m,
                y_m,
                radius_m,
                color_rgb: dream_color(rng, dream, [120, 160, 210]),
                emits_sound: false,
                spoken_text: None,
                charge_rate: 0.0,
            });
        }
    }
}

fn dream_position(
    config: &ScenarioConfig,
    rng: &mut StdRng,
    body: &BodySense,
    objects: &[SimObject],
    radius_m: f32,
    layout: DreamLayout,
    index: usize,
) -> Option<(f32, f32)> {
    for _ in 0..256 {
        let (x_m, y_m) = match layout {
            DreamLayout::Corridor => {
                let lane_y = if index % 2 == 0 {
                    config.arena.height_m * 0.25
                } else {
                    config.arena.height_m * 0.75
                };
                (
                    rng.gen_range(radius_m + 0.35..config.arena.width_m - radius_m - 0.35),
                    (lane_y + rng.gen_range(-0.35..0.35))
                        .clamp(radius_m + 0.35, config.arena.height_m - radius_m - 0.35),
                )
            }
            DreamLayout::ColumnField => {
                let cols = 4;
                let rows = 4;
                let col = index % cols;
                let row = (index / cols) % rows;
                let x = (col as f32 + 1.0) * config.arena.width_m / (cols as f32 + 1.0);
                let y = (row as f32 + 1.0) * config.arena.height_m / (rows as f32 + 1.0);
                (
                    (x + rng.gen_range(-0.35..0.35))
                        .clamp(radius_m + 0.35, config.arena.width_m - radius_m - 0.35),
                    (y + rng.gen_range(-0.35..0.35))
                        .clamp(radius_m + 0.35, config.arena.height_m - radius_m - 0.35),
                )
            }
            DreamLayout::Islands => {
                let center_x = if index % 3 == 0 {
                    config.arena.width_m * 0.30
                } else if index % 3 == 1 {
                    config.arena.width_m * 0.68
                } else {
                    config.arena.width_m * 0.50
                };
                let center_y = if index % 2 == 0 {
                    config.arena.height_m * 0.35
                } else {
                    config.arena.height_m * 0.70
                };
                (
                    (center_x + rng.gen_range(-0.9..0.9))
                        .clamp(radius_m + 0.35, config.arena.width_m - radius_m - 0.35),
                    (center_y + rng.gen_range(-0.9..0.9))
                        .clamp(radius_m + 0.35, config.arena.height_m - radius_m - 0.35),
                )
            }
            DreamLayout::Asymmetric => {
                let bias_left = rng.gen_bool(0.65);
                let x_max = if bias_left {
                    config.arena.width_m * 0.58
                } else {
                    config.arena.width_m - radius_m - 0.35
                };
                (
                    rng.gen_range(radius_m + 0.35..x_max.max(radius_m + 0.70)),
                    rng.gen_range(radius_m + 0.35..config.arena.height_m - radius_m - 0.35),
                )
            }
            DreamLayout::SmallRoom | DreamLayout::Room => (
                rng.gen_range(radius_m + 0.35..config.arena.width_m - radius_m - 0.35),
                rng.gen_range(radius_m + 0.35..config.arena.height_m - radius_m - 0.35),
            ),
        };
        if is_clear(x_m, y_m, radius_m, body, objects) {
            return Some((x_m, y_m));
        }
    }
    None
}

fn add_misleading_obstacle_near(
    config: &ScenarioConfig,
    rng: &mut StdRng,
    body: &BodySense,
    objects: &mut Vec<SimObject>,
    charger_x: f32,
    charger_y: f32,
    dream: DreamConfig,
) {
    let radius_m = rng.gen_range(0.16..0.28);
    for _ in 0..32 {
        let angle = rng.gen_range(-std::f32::consts::PI..std::f32::consts::PI);
        let distance = rng.gen_range(0.65..1.15);
        let x_m = (charger_x + angle.cos() * distance)
            .clamp(radius_m + 0.35, config.arena.width_m - radius_m - 0.35);
        let y_m = (charger_y + angle.sin() * distance)
            .clamp(radius_m + 0.35, config.arena.height_m - radius_m - 0.35);
        if is_clear(x_m, y_m, radius_m, body, objects) {
            let mut object = SimObject::obstacle(
                format!("dream-decoy-{}", objects.len()),
                choose(rng, DREAM_OBSTACLE_LABELS).to_string(),
                x_m,
                y_m,
                radius_m,
            );
            object.color_rgb = dream_color(rng, dream, [180, 90, 80]);
            objects.push(object);
            return;
        }
    }
}

pub fn mutate_dream_world(world: &mut VirtualWorld, rng: &mut StdRng, episode: u64) {
    let body = world.body();
    let arena = world.arena();
    let mut objects = world.objects();
    if objects.is_empty() {
        return;
    }
    let index = (episode as usize + rng.gen_range(0..objects.len())) % objects.len();
    let mut candidate = objects[index].clone();
    match &mut candidate.kind {
        SimObjectKind::Landmark { label } => {
            *label = choose(rng, DREAM_LANDMARK_LABELS).to_string();
            candidate.label = label.clone();
        }
        SimObjectKind::SoundSource { label } => {
            *label = choose(rng, DREAM_VOICE_LABELS).to_string();
            candidate.label = label.clone();
            candidate.spoken_text = Some(choose(rng, DREAM_VOICE_PHRASES).to_string());
        }
        SimObjectKind::Person { identity } => {
            let name = choose(rng, DREAM_PERSON_NAMES).to_string();
            *identity = Some(name.clone());
            candidate.label = name;
            candidate.spoken_text = Some(choose(rng, DREAM_PERSON_PHRASES).to_string());
        }
        SimObjectKind::Obstacle | SimObjectKind::Charger => {
            candidate.label = if matches!(candidate.kind, SimObjectKind::Charger) {
                choose(rng, DREAM_CHARGER_LABELS).to_string()
            } else {
                choose(rng, DREAM_OBSTACLE_LABELS).to_string()
            };
        }
    }

    let others = objects
        .iter()
        .enumerate()
        .filter_map(|(object_index, object)| (object_index != index).then_some(object.clone()))
        .collect::<Vec<_>>();
    let dx = rng.gen_range(-0.18..0.18);
    let dy = rng.gen_range(-0.18..0.18);
    let x_m = (candidate.x_m + dx).clamp(
        candidate.radius_m + 0.35,
        arena.width_m - candidate.radius_m - 0.35,
    );
    let y_m = (candidate.y_m + dy).clamp(
        candidate.radius_m + 0.35,
        arena.height_m - candidate.radius_m - 0.35,
    );
    if is_clear(x_m, y_m, candidate.radius_m, &body, &others) {
        candidate.x_m = x_m;
        candidate.y_m = y_m;
    }
    objects[index] = candidate;
    world.set_objects(objects);
}

fn dream_color(rng: &mut StdRng, dream: DreamConfig, base: [u8; 3]) -> [u8; 3] {
    if rng.gen_bool((0.25 + dream.weirdness * 0.65) as f64) {
        let palette = DREAM_COLORS[rng.gen_range(0..DREAM_COLORS.len())];
        [
            jitter_u8(rng, palette[0], 18),
            jitter_u8(rng, palette[1], 18),
            jitter_u8(rng, palette[2], 18),
        ]
    } else {
        [
            jitter_u8(rng, base[0], 24),
            jitter_u8(rng, base[1], 24),
            jitter_u8(rng, base[2], 24),
        ]
    }
}

fn jitter_u8(rng: &mut StdRng, value: u8, amount: i16) -> u8 {
    (value as i16 + rng.gen_range(-amount..=amount)).clamp(0, 255) as u8
}

fn choose<'a>(rng: &mut StdRng, values: &'a [&'a str]) -> &'a str {
    values[rng.gen_range(0..values.len())]
}

const DREAM_LANDMARK_LABELS: &[&str] = &[
    "blue arch",
    "red pillar",
    "glass tree",
    "sleeping door",
    "low moon",
    "paper tower",
    "mirror stone",
];

const DREAM_OBSTACLE_LABELS: &[&str] = &[
    "soft column",
    "warm wall",
    "crooked block",
    "black stump",
    "red shape",
    "quiet boulder",
];

const DREAM_CHARGER_LABELS: &[&str] = &[
    "green shrine",
    "bright nest",
    "charging flower",
    "silver dock",
];

const DREAM_PERSON_NAMES: &[&str] = &[
    "Mara", "Ivo", "Sable", "Noor", "Elian", "Vesper", "Lumen", "Orra",
];

const DREAM_PERSON_PHRASES: &[&str] = &[
    "The hallway keeps changing when you look away.",
    "Follow the warm light if you want to wake the charger.",
    "I remember this room from someone else's sleep.",
    "There is a safe path between the red shapes.",
    "Listen closely; the floor is telling us where to turn.",
    "The dream world is softer near the center.",
    "The mirror stone remembers every collision.",
    "Step around the low moon; it is heavier than it looks.",
];

const DREAM_VOICE_LABELS: &[&str] = &[
    "wall voice",
    "ceiling hum",
    "distant bell",
    "floor radio",
    "hidden choir",
];

const DREAM_VOICE_PHRASES: &[&str] = &[
    "A voice drifts through the walls.",
    "Something far away is humming your name.",
    "The room is not empty; it is waiting.",
    "The charger is singing behind the crooked block.",
    "Turn until the blue arch becomes small.",
];

const DREAM_COLORS: &[[u8; 3]] = &[
    [70, 180, 235],
    [230, 70, 95],
    [245, 210, 80],
    [160, 95, 220],
    [40, 220, 160],
    [235, 130, 45],
    [235, 235, 245],
];

fn dream_person_name(index: usize) -> &'static str {
    const NAMES: &[&str] = &["Mara", "Ivo", "Sable", "Noor", "Elian", "Vesper"];
    NAMES[index % NAMES.len()]
}

fn dream_person_phrase(index: usize) -> &'static str {
    const PHRASES: &[&str] = &[
        "The hallway keeps changing when you look away.",
        "Follow the warm light if you want to wake the charger.",
        "I remember this room from someone else's sleep.",
        "There is a safe path between the red shapes.",
        "Listen closely; the floor is telling us where to turn.",
        "The dream world is softer near the center.",
    ];
    PHRASES[index % PHRASES.len()]
}

fn dream_voice_phrase(index: usize) -> &'static str {
    const PHRASES: &[&str] = &[
        "A voice drifts through the walls.",
        "Something far away is humming your name.",
        "The room is not empty; it is waiting.",
    ];
    PHRASES[index % PHRASES.len()]
}

fn random_free_position(
    arena: ArenaConfig,
    rng: &mut StdRng,
    body: &BodySense,
    objects: &[SimObject],
    radius_m: f32,
) -> (f32, f32) {
    for _ in 0..256 {
        let x_m = rng.gen_range(radius_m + 0.35..arena.width_m - radius_m - 0.35);
        let y_m = rng.gen_range(radius_m + 0.35..arena.height_m - radius_m - 0.35);
        if is_clear(x_m, y_m, radius_m, body, objects) {
            return (x_m, y_m);
        }
    }
    (arena.width_m * 0.5, arena.height_m * 0.5)
}

fn is_clear(x_m: f32, y_m: f32, radius_m: f32, body: &BodySense, objects: &[SimObject]) -> bool {
    let dx = x_m - body.odometry.x_m;
    let dy = y_m - body.odometry.y_m;
    if (dx * dx + dy * dy).sqrt() < ROBOT_SPAWN_CLEARANCE_M + radius_m {
        return false;
    }
    objects.iter().all(|object| {
        let dx = x_m - object.x_m;
        let dy = y_m - object.y_m;
        (dx * dx + dy * dy).sqrt() >= radius_m + object.radius_m + 0.18
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use pete_sensors::World;

    fn configured_dream(seed: u64, dream: DreamConfig) -> ScenarioConfig {
        let mut config = ScenarioConfig::new(ScenarioKind::Dream, seed);
        config.dream = Some(dream);
        apply_dream_config(&mut config, dream);
        config.object_count = config.charger_count
            + config.obstacle_count
            + config.person_count
            + config.speaker_count
            + config.landmark_count;
        config
    }

    #[test]
    fn scenario_generation_is_deterministic_for_same_seed() {
        let left = build_scenario(ScenarioConfig::new(ScenarioKind::ChargerSeeking, 42));
        let right = build_scenario(ScenarioConfig::new(ScenarioKind::ChargerSeeking, 42));

        assert_eq!(left.metadata, right.metadata);
    }

    #[test]
    fn scenario_generation_differs_across_seeds() {
        let left = build_scenario(ScenarioConfig::new(ScenarioKind::MixedRoom, 42));
        let right = build_scenario(ScenarioConfig::new(ScenarioKind::MixedRoom, 43));

        assert_ne!(left.metadata, right.metadata);
    }

    #[test]
    fn dream_generation_is_deterministic_for_same_seed_and_config() {
        let dream = DreamConfig {
            seed: 42,
            weirdness: 0.55,
            density: 0.7,
            sociality: 0.5,
            hazard_bias: 0.35,
            charger_bias: 0.5,
        };
        let left = build_scenario(configured_dream(42, dream));
        let right = build_scenario(configured_dream(42, dream));

        assert_eq!(left.metadata, right.metadata);
    }

    #[test]
    fn dream_generation_differs_across_seeds() {
        let left = build_scenario(ScenarioConfig::new(ScenarioKind::Dream, 42));
        let right = build_scenario(ScenarioConfig::new(ScenarioKind::Dream, 43));

        assert_ne!(left.metadata, right.metadata);
    }

    #[test]
    fn objects_spawn_inside_arena_and_away_from_robot() {
        let scenario = build_scenario(ScenarioConfig::new(ScenarioKind::MixedRoom, 42));
        let body = &scenario.metadata.body;
        for object in &scenario.metadata.objects {
            assert!(object.x_m - object.radius_m >= 0.0);
            assert!(object.y_m - object.radius_m >= 0.0);
            assert!(object.x_m + object.radius_m <= scenario.metadata.arena.width_m);
            assert!(object.y_m + object.radius_m <= scenario.metadata.arena.height_m);
            let dx = object.x_m - body.odometry.x_m;
            let dy = object.y_m - body.odometry.y_m;
            assert!((dx * dx + dy * dy).sqrt() >= ROBOT_SPAWN_CLEARANCE_M);
        }
    }

    #[test]
    fn dream_objects_spawn_inside_arena_and_clear() {
        let dream = DreamConfig {
            seed: 99,
            weirdness: 0.8,
            density: 0.8,
            sociality: 0.6,
            hazard_bias: 0.5,
            charger_bias: 0.7,
        };
        let scenario = build_scenario(configured_dream(99, dream));
        let body = &scenario.metadata.body;
        for (index, object) in scenario.metadata.objects.iter().enumerate() {
            assert!(object.x_m - object.radius_m >= 0.0, "{object:?}");
            assert!(object.y_m - object.radius_m >= 0.0, "{object:?}");
            assert!(
                object.x_m + object.radius_m <= scenario.metadata.arena.width_m,
                "{object:?}"
            );
            assert!(
                object.y_m + object.radius_m <= scenario.metadata.arena.height_m,
                "{object:?}"
            );
            let dx = object.x_m - body.odometry.x_m;
            let dy = object.y_m - body.odometry.y_m;
            assert!(
                (dx * dx + dy * dy).sqrt() >= ROBOT_SPAWN_CLEARANCE_M + object.radius_m,
                "{object:?}"
            );

            for other in scenario.metadata.objects.iter().skip(index + 1) {
                let dx = object.x_m - other.x_m;
                let dy = object.y_m - other.y_m;
                assert!(
                    (dx * dx + dy * dy).sqrt() >= object.radius_m + other.radius_m + 0.18,
                    "{object:?} overlaps {other:?}"
                );
            }
        }
    }

    #[test]
    fn dream_contains_meaningful_non_obstacle_when_biases_allow() {
        let dream = DreamConfig {
            seed: 123,
            weirdness: 0.45,
            density: 0.75,
            sociality: 0.75,
            hazard_bias: 0.25,
            charger_bias: 0.9,
        };
        let scenario = build_scenario(configured_dream(123, dream));

        assert!(scenario.metadata.objects.iter().any(|object| matches!(
            object.kind,
            SimObjectKind::Charger
                | SimObjectKind::Person { .. }
                | SimObjectKind::SoundSource { .. }
                | SimObjectKind::Landmark { .. }
        )));
    }

    #[test]
    fn high_weirdness_produces_sensor_useful_strangeness() {
        let dream = DreamConfig {
            seed: 777,
            weirdness: 1.0,
            density: 0.75,
            sociality: 0.5,
            hazard_bias: 0.35,
            charger_bias: 0.5,
        };
        let scenario = build_scenario(configured_dream(777, dream));
        let has_strange_projection = scenario.metadata.objects.iter().any(|object| {
            matches!(
                object.kind,
                SimObjectKind::Landmark { .. } | SimObjectKind::SoundSource { .. }
            ) || DREAM_LANDMARK_LABELS.contains(&object.label.as_str())
                || DREAM_OBSTACLE_LABELS.contains(&object.label.as_str())
                || DREAM_CHARGER_LABELS.contains(&object.label.as_str())
                || !matches!(
                    object.color_rgb,
                    [180, 90, 80] | [80, 220, 130] | [220, 180, 140] | [80, 80, 220]
                )
        });

        assert!(has_strange_projection);
    }

    #[test]
    fn charger_scenario_contains_at_least_one_charger() {
        let scenario = build_scenario(ScenarioConfig::new(ScenarioKind::ChargerSeeking, 7));

        assert!(scenario
            .metadata
            .objects
            .iter()
            .any(|object| matches!(object.kind, SimObjectKind::Charger)));
    }

    #[test]
    fn room_scenarios_contain_a_charger() {
        for kind in [
            ScenarioKind::EmptyRoom,
            ScenarioKind::ObstacleAvoidance,
            ScenarioKind::CornerTrap,
            ScenarioKind::ColumnTrap,
            ScenarioKind::PersonAndSpeaker,
            ScenarioKind::MixedRoom,
        ] {
            let scenario = build_scenario(ScenarioConfig::new(kind, 7));

            assert!(
                scenario
                    .metadata
                    .objects
                    .iter()
                    .any(|object| matches!(object.kind, SimObjectKind::Charger)),
                "{kind:?} should include a charger"
            );
        }
    }

    #[tokio::test]
    async fn obstacle_scenario_has_near_obstacle_frame() {
        let mut scenario = build_scenario(ScenarioConfig::new(ScenarioKind::ObstacleAvoidance, 7));
        let snapshot = scenario.world.snapshot().await.unwrap();

        assert!(snapshot.range.nearest_m.unwrap_or(10.0) < 0.5);
    }

    #[tokio::test]
    async fn corner_trap_scenario_starts_constrained() {
        let mut scenario = build_scenario(ScenarioConfig::new(ScenarioKind::CornerTrap, 7));
        let snapshot = scenario.world.snapshot().await.unwrap();

        assert!(snapshot.range.nearest_m.unwrap_or(10.0) < 0.35);
    }

    #[tokio::test]
    async fn person_speaker_scenario_projects_social_senses() {
        let mut scenario = build_scenario(ScenarioConfig::new(ScenarioKind::PersonAndSpeaker, 7));
        let snapshot = scenario.world.snapshot().await.unwrap();

        assert!(snapshot.eye_frame.is_some());
        assert!(!snapshot.eye.frames.is_empty());
        assert!(!snapshot.ear.features.is_empty());
        let transcript = snapshot.ear.transcript.as_deref().unwrap_or_default();
        assert!(transcript.contains("Mara says"));
        assert!(!transcript.contains("person 0 sound"));
        assert!(snapshot.ear_pcm.is_some());
        assert!(!snapshot.voice.embeddings.is_empty());
        assert!(!snapshot.face.embeddings.is_empty());
        assert!(!snapshot.kinect.color_features.is_empty());
        assert!(!snapshot.kinect.depth_m.is_empty());
        assert!(!snapshot.kinect.skeletons.is_empty());
    }
}
