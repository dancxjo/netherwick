
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
        ScenarioKind::ConcaveTrap,
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
async fn concave_trap_scenario_starts_constrained() {
    let mut scenario = build_scenario(ScenarioConfig::new(ScenarioKind::ConcaveTrap, 7));
    let snapshot = scenario.world.snapshot().await.unwrap();

    assert!(snapshot.range.nearest_m.unwrap_or(10.0) < 0.7);
    assert_eq!(
        scenario
            .metadata
            .objects
            .iter()
            .filter(|object| matches!(object.kind, SimObjectKind::Obstacle))
            .count(),
        5
    );
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
    assert!(!snapshot.voice.vectors.is_empty());
    assert!(!snapshot.face.vectors.is_empty());
    assert!(!snapshot.kinect.color_features.is_empty());
    assert!(!snapshot.kinect.depth_m.is_empty());
    assert!(!snapshot.kinect.skeletons.is_empty());
}
