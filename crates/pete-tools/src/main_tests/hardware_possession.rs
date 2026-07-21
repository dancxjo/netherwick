#[tokio::test]
async fn hardware_env_report_has_expected_shape() {
    let report = collect_hardware_env_report().await;
    assert!(report.get("os").is_some());
    assert!(report.get("architecture").is_some());
    assert!(report.get("serial_devices").unwrap().is_array());
    assert!(report.get("gps_serial_candidates").unwrap().is_array());
    assert_eq!(report["default_gps"]["baud"].as_u64(), Some(9600));
    assert!(report.get("lidar_serial_candidates").unwrap().is_array());
    assert_eq!(
        report["default_lidar"]["baud"].as_u64(),
        Some(u64::from(Lfcd2SenseProvider::BAUD_RATE))
    );
    assert!(report.get("i2c_devices").unwrap().is_array());
    assert_eq!(
        report["default_imu"]["device"].as_str(),
        Some(DEFAULT_MPU6050_IMU_DEVICE)
    );
    assert!(report.get("camera_devices").unwrap().is_array());
    assert!(report.get("audio_input_devices").unwrap().is_array());
    assert!(report.get("kinect").unwrap().is_object());
    assert!(report.get("data_dirs_writable").unwrap().is_object());
}

#[test]
fn imu_device_is_registered_only_after_discovery_or_explicit_configuration() {
    assert_eq!(
        selected_imu_device(None, false),
        discover_local_imu_device()
    );
    assert_eq!(selected_imu_device(None, true), None);
    assert_eq!(
        selected_imu_device(Some("/dev/i2c-1@0x69"), true),
        Some("/dev/i2c-1@0x69")
    );
    assert_eq!(selected_imu_device(Some("none"), false), None);
}

#[test]
fn imu_source_overrides_never_start_a_duplicate_local_provider() {
    let Command::Robot(brainstem) = Cli::try_parse_from([
        "pete",
        "robot",
        "--imu-source",
        "brainstem",
        "--imu",
        "/dev/i2c-1",
    ])
    .unwrap()
    .command
    else {
        panic!("expected robot command");
    };
    assert!(!local_imu_provider_allowed(&brainstem));
    assert!(matches!(
        imu_source_override(&brainstem),
        ImuSourceOverride::Force(ref source) if source == "brainstem_board_imu"
    ));

    let Command::Robot(disabled) = Cli::try_parse_from(["pete", "robot", "--imu-source", "none"])
        .unwrap()
        .command
    else {
        panic!("expected robot command");
    };
    assert!(!local_imu_provider_allowed(&disabled));
    assert!(matches!(
        imu_source_override(&disabled),
        ImuSourceOverride::Disabled
    ));
}

#[test]
fn serial_auto_selection_keeps_lidar_gps_and_create_separate() {
    let report = serde_json::json!({
        "serial_devices": [
            "/dev/serial/by-id/usb-ROBOTIS_USB2LDS_LDS-01",
            "/dev/serial/by-id/usb-u-blox_AG_-_www.u-blox.com_u-blox_7-if00",
            "/dev/ttyACM0",
            "/dev/ttyUSB0"
        ]
    });

    let lidar = selected_lidar_device(None, false, &report);
    assert_eq!(
        lidar.as_deref(),
        Some("/dev/serial/by-id/usb-ROBOTIS_USB2LDS_LDS-01")
    );

    assert_eq!(
        selected_create_port("auto", &report, lidar.as_deref()),
        Some("/dev/ttyUSB0".to_string())
    );
    assert_eq!(
        selected_gps_device(None, false, &report, Some("/dev/ttyUSB0")),
        Some("/dev/serial/by-id/usb-u-blox_AG_-_www.u-blox.com_u-blox_7-if00".to_string())
    );
    assert_eq!(
        selected_gps_device(Some("/dev/ttyACM1"), false, &report, Some("/dev/ttyUSB0")),
        Some("/dev/ttyACM1".to_string())
    );
    assert_eq!(
        selected_gps_device(Some("none"), false, &report, Some("/dev/ttyUSB0")),
        None
    );
    assert_eq!(selected_lidar_device(Some("none"), false, &report), None);
    assert_eq!(
        selected_lidar_device(Some("/dev/ttyUSB9"), true, &report),
        Some("/dev/ttyUSB9".to_string())
    );
}

#[test]
fn local_cockpit_uses_the_rpi5_brainstem_address_not_a_serial_device() {
    let report = serde_json::json!({
        "serial_devices": ["/dev/ttyUSB0"]
    });
    let address = "127.0.0.1:9876".parse().unwrap();
    assert_eq!(
        selected_cockpit_endpoint(
            CockpitBackendArg::Local,
            "auto",
            "192.168.4.1:80",
            address,
            &report,
            None,
        ),
        Some(address.to_string())
    );
}

#[tokio::test]
async fn possession_mode_never_falls_back_when_brainstem_is_missing() {
    let result = open_robot_cockpit_or_fallback(
        CockpitBackendArg::Uart,
        None,
        RobotMode::Slow,
        None,
        None,
        50,
        500,
    );
    let error = match result {
        Ok(_) => panic!("possession unexpectedly fell back"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("stable brainstem USB CDC"));
}

#[tokio::test]
async fn physical_capture_uses_runtime_frame_time_when_body_time_is_stale() {
    let temp_dir = temp_path("pete_physical_capture_runtime_time");
    let mut writer = CaptureWriter::create(&temp_dir, CaptureSource::RealRobot, Some(100))
        .await
        .unwrap();
    let mut snapshot = WorldSnapshot::default();
    snapshot.body.last_update_ms = 100;
    let mut tick = tick_with_action(ActionPrimitive::Stop);
    tick.frame.t_ms = 250;
    tick.frame.now.t_ms = 250;
    tick.frame.now.body.last_update_ms = 100;

    append_real_robot_snapshot(&mut writer, &snapshot, &tick)
        .await
        .unwrap();
    writer.finish().await.unwrap();

    let frames = CaptureReader::open(&temp_dir)
        .await
        .unwrap()
        .read_frames()
        .await
        .unwrap();
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].t_ms, 250);
    assert_eq!(frames[0].snapshot.body.last_update_ms, 100);
    let _ = fs::remove_dir_all(&temp_dir);
}

#[test]
fn possession_reconnect_backoff_is_exponential_and_bounded() {
    assert_eq!(next_reconnect_backoff_ms(250, 5_000), 500);
    assert_eq!(next_reconnect_backoff_ms(4_000, 5_000), 5_000);
    assert_eq!(next_reconnect_backoff_ms(5_000, 5_000), 5_000);
}

struct FreshPacketCockpit {
    status_reads: usize,
    stopped: bool,
    stream_requested: bool,
}

impl Cockpit for FreshPacketCockpit {
    fn execute(
        &mut self,
        request: pete_cockpit::CockpitRequest,
    ) -> pete_cockpit::Result<pete_cockpit::CockpitResponse> {
        match request {
            pete_cockpit::CockpitRequest::Stop => {
                self.stopped = true;
                Ok(pete_cockpit::CockpitResponse::Accepted)
            }
            pete_cockpit::CockpitRequest::StreamSensors {
                enabled: true,
                packet_id: 0,
                ..
            } => {
                assert!(self.stopped, "stream requested before STOP");
                self.stream_requested = true;
                Ok(pete_cockpit::CockpitResponse::Accepted)
            }
            pete_cockpit::CockpitRequest::GetStatus => {
                self.status_reads += 1;
                let fresh = self.stream_requested && self.status_reads >= 3;
                let count = if fresh { 2 } else { 1 };
                let packet_ms = if fresh { 995 } else { 100 };
                Ok(pete_cockpit::CockpitResponse::Status(
                    pete_cockpit::CockpitStatus {
                        raw: serde_json::json!({
                            "uptime_ms": 1_000,
                            "current_runtime_state": "idle",
                            "current_command": "stop",
                            "create_sensors": {
                                "last_packet_id": 0,
                                "complete_packet_count": count,
                                "last_complete_packet_timestamp_ms": packet_ms
                            }
                        })
                        .to_string(),
                    },
                ))
            }
            other => panic!("unexpected readiness request: {other:?}"),
        }
    }

    fn handshake(
        &mut self,
        _hello: HandshakeHello,
    ) -> pete_cockpit::Result<pete_cockpit::HandshakeOutcome> {
        Err(CockpitError::Policy("not used by readiness test".into()))
    }

    fn execute_in_session(
        &mut self,
        _session: &pete_cockpit::CockpitSession,
        request: pete_cockpit::CockpitRequest,
    ) -> pete_cockpit::Result<pete_cockpit::CockpitResponse> {
        self.execute(request)
    }

    fn execute_with_lease(
        &mut self,
        _session: &pete_cockpit::CockpitSession,
        _lease: &pete_cockpit::ControlLease,
        request: pete_cockpit::CockpitRequest,
    ) -> pete_cockpit::Result<pete_cockpit::CockpitResponse> {
        self.execute(request)
    }

    fn execute_with_service_lease(
        &mut self,
        _session: &pete_cockpit::CockpitSession,
        _lease: &pete_cockpit::ServiceLease,
        _request: pete_cockpit::CockpitRequest,
    ) -> pete_cockpit::Result<pete_cockpit::CockpitResponse> {
        Err(CockpitError::Policy("service mode unavailable".into()))
    }
}

#[test]
fn reconnect_readiness_requires_stop_and_new_complete_packet() {
    let mut cockpit = FreshPacketCockpit {
        status_reads: 0,
        stopped: false,
        stream_requested: false,
    };

    establish_create_sensor_stream(&mut cockpit, true).unwrap();

    assert!(cockpit.stopped);
    assert!(cockpit.stream_requested);
    assert!(cockpit.status_reads >= 3);
}

struct DropTrackedCockpit {
    drops: Arc<AtomicUsize>,
}

impl Drop for DropTrackedCockpit {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

impl Cockpit for DropTrackedCockpit {
    fn execute(
        &mut self,
        _request: pete_cockpit::CockpitRequest,
    ) -> pete_cockpit::Result<pete_cockpit::CockpitResponse> {
        Err(CockpitError::Policy("test cockpit is closed".into()))
    }

    fn handshake(
        &mut self,
        _hello: HandshakeHello,
    ) -> pete_cockpit::Result<pete_cockpit::HandshakeOutcome> {
        Err(CockpitError::Policy("test cockpit is closed".into()))
    }

    fn execute_in_session(
        &mut self,
        _session: &pete_cockpit::CockpitSession,
        _request: pete_cockpit::CockpitRequest,
    ) -> pete_cockpit::Result<pete_cockpit::CockpitResponse> {
        Err(CockpitError::Policy("test cockpit is closed".into()))
    }

    fn execute_with_lease(
        &mut self,
        _session: &pete_cockpit::CockpitSession,
        _lease: &pete_cockpit::ControlLease,
        _request: pete_cockpit::CockpitRequest,
    ) -> pete_cockpit::Result<pete_cockpit::CockpitResponse> {
        Err(CockpitError::Policy("test cockpit is closed".into()))
    }

    fn execute_with_service_lease(
        &mut self,
        _session: &pete_cockpit::CockpitSession,
        _lease: &pete_cockpit::ServiceLease,
        _request: pete_cockpit::CockpitRequest,
    ) -> pete_cockpit::Result<pete_cockpit::CockpitResponse> {
        Err(CockpitError::Policy("test cockpit is closed".into()))
    }
}

#[test]
fn possession_reconnect_drops_existing_cockpit_before_opening_replacement() {
    let drops = Arc::new(AtomicUsize::new(0));
    let cockpit: Box<dyn Cockpit + Send> = Box::new(DropTrackedCockpit {
        drops: Arc::clone(&drops),
    });
    let mut safe = SafeCockpit::new(cockpit);

    disconnect_possession_cockpit_for_reconnect(&mut safe);

    assert_eq!(drops.load(Ordering::SeqCst), 1);
    let error = safe.client_mut().stop().unwrap_err();
    assert!(error.to_string().contains("reconnect in progress"));
}

struct BusyShutdownCockpit {
    stop_busy_remaining: usize,
    exorcize_busy_remaining: usize,
    stop_attempts: usize,
    exorcize_attempts: usize,
    stopped: bool,
    exorcized: bool,
}

impl BusyShutdownCockpit {
    fn busy(command_id: u32) -> CockpitError {
        CockpitError::Rejected {
            command_id,
            reason: "busy".into(),
        }
    }
}

impl Cockpit for BusyShutdownCockpit {
    fn exorcize(&mut self) -> pete_cockpit::Result<()> {
        self.exorcize_attempts += 1;
        if self.exorcize_busy_remaining > 0 {
            self.exorcize_busy_remaining -= 1;
            return Err(Self::busy(100 + self.exorcize_attempts as u32));
        }
        self.exorcized = true;
        Ok(())
    }

    fn execute(
        &mut self,
        request: pete_cockpit::CockpitRequest,
    ) -> pete_cockpit::Result<pete_cockpit::CockpitResponse> {
        match request {
            pete_cockpit::CockpitRequest::Stop => {
                self.stop_attempts += 1;
                if self.stop_busy_remaining > 0 {
                    self.stop_busy_remaining -= 1;
                    return Err(Self::busy(self.stop_attempts as u32));
                }
                self.stopped = true;
                Ok(pete_cockpit::CockpitResponse::Accepted)
            }
            other => panic!("unexpected shutdown request: {other:?}"),
        }
    }

    fn handshake(
        &mut self,
        _hello: HandshakeHello,
    ) -> pete_cockpit::Result<pete_cockpit::HandshakeOutcome> {
        Err(CockpitError::Policy("not used by shutdown test".into()))
    }

    fn execute_in_session(
        &mut self,
        _session: &pete_cockpit::CockpitSession,
        request: pete_cockpit::CockpitRequest,
    ) -> pete_cockpit::Result<pete_cockpit::CockpitResponse> {
        self.execute(request)
    }

    fn execute_with_lease(
        &mut self,
        _session: &pete_cockpit::CockpitSession,
        _lease: &pete_cockpit::ControlLease,
        request: pete_cockpit::CockpitRequest,
    ) -> pete_cockpit::Result<pete_cockpit::CockpitResponse> {
        self.execute(request)
    }

    fn execute_with_service_lease(
        &mut self,
        _session: &pete_cockpit::CockpitSession,
        _lease: &pete_cockpit::ServiceLease,
        _request: pete_cockpit::CockpitRequest,
    ) -> pete_cockpit::Result<pete_cockpit::CockpitResponse> {
        Err(CockpitError::Policy("service mode unavailable".into()))
    }
}

#[test]
fn possession_shutdown_retries_plain_busy_stop_and_exorcize() {
    let mut cockpit = BusyShutdownCockpit {
        stop_busy_remaining: 2,
        exorcize_busy_remaining: 1,
        stop_attempts: 0,
        exorcize_attempts: 0,
        stopped: false,
        exorcized: false,
    };

    run_possession_shutdown_with_retry(&mut cockpit, 5, Duration::ZERO).unwrap();

    assert!(cockpit.stopped);
    assert!(cockpit.exorcized);
    assert_eq!(cockpit.stop_attempts, 3);
    assert_eq!(cockpit.exorcize_attempts, 2);
}

#[test]
fn simulated_possession_reconnect_gets_fresh_session_and_lease() {
    let (mut first, _, _) = open_robot_cockpit_or_fallback(
        CockpitBackendArg::Sim,
        Some("mock"),
        RobotMode::Slow,
        None,
        None,
        50,
        500,
    )
    .unwrap();
    let first_snapshot = first.possession_snapshot().unwrap();
    assert!(first_snapshot.lease_remaining_ms > 59_000);
    first
        .cmd_vel(50, 0, 30_000)
        .expect("first lease applies bounded motion");
    drop(first);

    let (mut second, _, _) = open_robot_cockpit_or_fallback(
        CockpitBackendArg::Sim,
        Some("mock"),
        RobotMode::Slow,
        None,
        None,
        50,
        500,
    )
    .unwrap();
    let second_snapshot = second.possession_snapshot().unwrap();
    assert_ne!(first_snapshot.session_id, second_snapshot.session_id);
    assert_ne!(first_snapshot.lease_id, second_snapshot.lease_id);
    assert!(second_snapshot.possessed);
    assert_eq!(
        second.get_status().unwrap().summary().active_motion,
        Some(false)
    );
}
