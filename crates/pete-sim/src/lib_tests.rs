
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
        .apply_motion(MotionCommand::Forward { speed_m_s: 1.0 })
        .unwrap();

    assert!(body.odometry.x_m >= 0.2 - f32::EPSILON);
    assert!(body.flags.bump_left && body.flags.bump_right);
    assert!(body.flags.wall);
}

#[tokio::test]
async fn simulator_imu_orientation_is_roll_pitch_yaw() {
    let (mut world, _motor) = VirtualWorld::new_with_motor(0, arena());
    {
        let mut guard = world.state.lock().unwrap();
        guard.snapshot.body.odometry.heading_rad = 1.25;
    }

    let snapshot = world.snapshot().await.unwrap();

    assert_eq!(snapshot.imu.schema_version, 1);
    assert_eq!(snapshot.imu.orientation, vec![0.0, 0.0, 1.25]);
    assert_eq!(snapshot.imu.acceleration.len(), 3);
    assert_eq!(snapshot.imu.angular_velocity.len(), 3);
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
        .apply_motion(MotionCommand::Forward { speed_m_s: 2.0 })
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

    let body = motor.apply_motion(MotionCommand::Stop).unwrap();

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

    assert_eq!(snapshot.face.vectors.len(), 1);
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
    assert_eq!(snapshot.voice.vectors.len(), 1);
}
