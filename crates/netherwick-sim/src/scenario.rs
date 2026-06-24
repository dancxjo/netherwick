use netherwick_body::BodySense;
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::{ArenaConfig, SimMotorComplex, SimObject, SimObjectKind, VirtualWorld};

pub const ROBOT_SPAWN_CLEARANCE_M: f32 = 0.45;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScenarioKind {
    EmptyRoom,
    ObstacleAvoidance,
    ChargerSeeking,
    PersonAndSpeaker,
    MixedRoom,
}

impl ScenarioKind {
    pub fn slug(self) -> &'static str {
        match self {
            Self::EmptyRoom => "empty-room",
            Self::ObstacleAvoidance => "obstacle-avoidance",
            Self::ChargerSeeking => "charger-seeking",
            Self::PersonAndSpeaker => "person-speaker-room",
            Self::MixedRoom => "mixed-room",
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
        };
        match kind {
            ScenarioKind::EmptyRoom => {}
            ScenarioKind::ObstacleAvoidance => {
                config.obstacle_count = 7;
            }
            ScenarioKind::ChargerSeeking => {
                config.charger_count = 2;
                config.obstacle_count = 4;
            }
            ScenarioKind::PersonAndSpeaker => {
                config.person_count = 1;
                config.speaker_count = 1;
                config.obstacle_count = 2;
            }
            ScenarioKind::MixedRoom => {
                config.charger_count = 1;
                config.obstacle_count = 5;
                config.person_count = 1;
                config.speaker_count = 1;
            }
        }
        config.object_count = config.charger_count
            + config.obstacle_count
            + config.person_count
            + config.speaker_count;
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
    pub motors: SimMotorComplex,
    pub metadata: ScenarioMetadata,
}

pub fn default_sim_world(seed: u64) -> (VirtualWorld, SimMotorComplex) {
    let scenario = build_scenario(ScenarioConfig::new(ScenarioKind::MixedRoom, seed));
    (scenario.world, scenario.motors)
}

pub fn build_scenario(config: ScenarioConfig) -> ScenarioWorld {
    let mut rng = StdRng::seed_from_u64(config.seed);
    let (mut world, motors) = VirtualWorld::new_with_motor(config.seed, config.arena);
    let body = spawn_body(&config, &mut rng);
    world.set_body(body.clone());

    let mut objects = Vec::new();
    add_kind_objects(&config, &mut rng, &body, &mut objects);
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
        } else {
            random_free_position(config.arena, rng, body, objects, 0.35)
        };
        objects.push(SimObject::obstacle(
            format!("obstacle-{index}"),
            format!("obstacle {index}"),
            x_m,
            y_m,
            rng.gen_range(0.22..0.42),
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
        objects.push(SimObject {
            id: format!("person-{index}"),
            label: format!("person {index}"),
            kind: SimObjectKind::Person {
                identity: Some(format!("sim-person-{index}")),
            },
            x_m,
            y_m,
            radius_m: 0.22,
            color_rgb: [220, 180, 140],
            emits_sound: false,
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
            charge_rate: 0.0,
        });
    }
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
    use netherwick_sensors::World;

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
    fn charger_scenario_contains_at_least_one_charger() {
        let scenario = build_scenario(ScenarioConfig::new(ScenarioKind::ChargerSeeking, 7));

        assert!(scenario
            .metadata
            .objects
            .iter()
            .any(|object| matches!(object.kind, SimObjectKind::Charger)));
    }

    #[tokio::test]
    async fn obstacle_scenario_has_near_obstacle_frame() {
        let mut scenario = build_scenario(ScenarioConfig::new(ScenarioKind::ObstacleAvoidance, 7));
        let snapshot = scenario.world.snapshot().await.unwrap();

        assert!(snapshot.range.nearest_m.unwrap_or(10.0) < 0.5);
    }

    #[tokio::test]
    async fn person_speaker_scenario_projects_social_senses() {
        let mut scenario = build_scenario(ScenarioConfig::new(ScenarioKind::PersonAndSpeaker, 7));
        let snapshot = scenario.world.snapshot().await.unwrap();

        assert!(snapshot.eye_frame.is_some());
        assert!(!snapshot.eye.frames.is_empty());
        assert!(!snapshot.ear.features.is_empty());
        assert!(snapshot.ear_pcm.is_some());
        assert!(!snapshot.voice.embeddings.is_empty());
        assert!(!snapshot.face.embeddings.is_empty());
        assert!(!snapshot.kinect.color_features.is_empty());
        assert!(!snapshot.kinect.depth_m.is_empty());
        assert!(!snapshot.kinect.skeletons.is_empty());
    }
}
