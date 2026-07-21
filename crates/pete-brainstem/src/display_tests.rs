    use super::*;

    fn normal_status() -> DisplayStatus {
        DisplayStatus {
            runtime_state: RuntimeState::Idle as u8,
            body_state: BodyState::Idle as u8,
            create_power_state: 2,
            oi_mode: 3,
            oi_seen: true,
            oi_fresh: true,
            authority_active: false,
            imu_enabled: true,
            imu_health: ImuHealthCode::Ok as u8,
            last_error: 0,
            wifi_state: WIFI_SERVICES_STARTED,
            network: DisplayNetwork {
                ssid_suffix: Some(1_337_420),
                active_leases: 0,
            },
            battery: Some(BatteryStatus {
                percent: 73,
                charging: false,
            }),
            battery_stale: false,
        }
    }

    fn no_safety() -> DisplaySafety {
        DisplaySafety {
            estop_latched: false,
            safety_latch_kind: None,
        }
    }

    fn assert_lines(actual: DisplayPage, line1: &str, line2: &str) {
        assert_eq!(actual.line1.as_str(), line1);
        assert_eq!(actual.line2.as_str(), line2);
    }

    #[test]
    fn normal_pages_prioritize_large_status_and_real_battery() {
        let status = normal_status();
        assert_lines(status.page(no_safety(), 0), "PETE  READY", "CTRL OPEN");
        assert_lines(
            status.page(no_safety(), PAGE_ROTATION_MS),
            "PETE  READY",
            "CTRL OPEN",
        );
        assert_lines(
            status.page(no_safety(), PAGE_ROTATION_MS * 2),
            "BATT 73%",
            "ON BATTERY",
        );
    }

    #[test]
    fn active_control_authority_replaces_the_normal_imu_cell() {
        let mut status = normal_status();
        status.authority_active = true;
        assert_lines(status.page(no_safety(), 0), "PETE  READY", "CTRL ACTIVE");
    }

    #[test]
    fn network_failure_rotates_as_secondary_instead_of_monopolizing() {
        let mut status = normal_status();
        status.wifi_state = WIFI_ERROR;
        assert_lines(status.page(no_safety(), 0), "PETE  READY", "CTRL OPEN");
        assert_lines(
            status.page(no_safety(), PAGE_ROTATION_MS),
            "pete-snyk",
            "192.168.4.1 ERROR",
        );
        assert_lines(
            status.page(no_safety(), PAGE_ROTATION_MS * 2),
            "BATT 73%",
            "ON BATTERY",
        );
    }

    #[test]
    fn charging_is_a_persistent_positive_page() {
        let mut status = normal_status();
        status.battery = Some(BatteryStatus {
            percent: 73,
            charging: true,
        });
        for now_ms in [0, PAGE_ROTATION_MS, PAGE_ROTATION_MS * 2] {
            assert_lines(status.page(no_safety(), now_ms), "BATT 73%", "CHARGING");
        }
    }

    #[test]
    fn safety_and_fault_pages_have_stable_priority() {
        let mut status = normal_status();
        status.oi_fresh = false;
        status.battery = Some(BatteryStatus {
            percent: 10,
            charging: true,
        });
        status.imu_health = ImuHealthCode::Absent as u8;

        assert_lines(
            status.page(
                DisplaySafety {
                    estop_latched: true,
                    safety_latch_kind: Some(SafetyEventKind::Tilt),
                },
                0,
            ),
            "ESTOP",
            "",
        );
        assert_lines(
            status.page(
                DisplaySafety {
                    estop_latched: false,
                    safety_latch_kind: Some(SafetyEventKind::Impact),
                },
                0,
            ),
            "IMPACT",
            "",
        );
        assert_lines(status.page(no_safety(), 0), "OI LINK", "LOST");
    }

    #[test]
    fn invalid_or_missing_battery_never_creates_a_battery_page() {
        let mut status = normal_status();
        status.battery = None;
        assert_lines(
            status.page(no_safety(), PAGE_ROTATION_MS * 2),
            "PETE  READY",
            "CTRL OPEN",
        );
    }

    #[test]
    fn stale_battery_uses_a_full_alert_after_a_jitter_tolerant_window() {
        let mut snapshot = status::snapshot(10_000);
        snapshot.uptime_ms = 10_000;
        snapshot.create_sensor_complete_packet_count = 1;
        snapshot.create_sensor_capacity_mah = 100;
        snapshot.create_sensor_charge_mah = 10;
        snapshot.create_sensor_last_complete_packet_timestamp_ms =
            snapshot.uptime_ms - BATTERY_FRESHNESS_MS + 1;
        let network = DisplayNetwork {
            ssid_suffix: None,
            active_leases: 0,
        };

        let fresh = DisplayStatus::from_snapshot(&snapshot, network);
        assert_eq!(fresh.battery.map(|battery| battery.percent), Some(10));
        assert!(!fresh.battery_stale);

        snapshot.create_sensor_last_complete_packet_timestamp_ms =
            snapshot.uptime_ms - BATTERY_FRESHNESS_MS - 1;
        let stale = DisplayStatus::from_snapshot(&snapshot, network);
        assert_eq!(stale.battery, None);
        assert!(stale.battery_stale);
        assert_lines(stale.page(no_safety(), 0), "BATT", "STALE");
    }

    #[test]
    fn low_battery_and_offline_imu_use_existing_health_conditions() {
        let mut status = normal_status();
        status.battery = Some(BatteryStatus {
            percent: LOW_BATTERY_PERCENT as u8,
            charging: false,
        });
        assert_lines(status.page(no_safety(), 0), "LOW", "BATT");

        status.battery = Some(BatteryStatus {
            percent: 21,
            charging: false,
        });
        status.imu_health = ImuHealthCode::Fault as u8;
        assert_lines(status.page(no_safety(), 0), "IMU", "OFFLINE");
    }

    #[test]
    fn moving_and_passive_states_render_run_and_stop() {
        let mut status = normal_status();
        status.body_state = BodyState::Moving as u8;
        assert_lines(status.page(no_safety(), 0), "PETE  RUN", "CTRL OPEN");

        status.body_state = BodyState::Idle as u8;
        status.oi_mode = 1;
        assert_lines(status.page(no_safety(), 0), "PETE  STOP", "CTRL OPEN");
    }

    #[test]
    fn startup_create_and_runtime_diagnostics_are_explicit() {
        let mut status = normal_status();
        status.oi_seen = false;
        status.oi_fresh = false;
        status.body_state = BodyState::WaitingForCreate as u8;
        assert_lines(
            status.page(no_safety(), 0),
            "pete-snyk",
            "192.168.4.1 READY",
        );
        assert_lines(status.page(no_safety(), PAGE_ROTATION_MS), "WAIT", "CREATE");

        status.create_power_state = CREATE_POWER_OFF;
        assert_lines(status.page(no_safety(), 0), "POWER", "OFF");

        status.create_power_state = 2;
        status.runtime_state = RuntimeState::Error as u8;
        status.body_state = BodyState::Error as u8;
        for (error, expected) in [
            (ERROR_CREATE_NO_RESPONSE, ("OI NO", "RX")),
            (ERROR_UART_FRAMING, ("UART", "FRAME")),
            (ERROR_TIMEOUT, ("TIME", "OUT")),
            (ERROR_INVALID_PACKET, ("BAD", "PACKET")),
            (0, ("RUNTIME", "ERROR")),
        ] {
            status.last_error = error;
            assert_lines(status.page(no_safety(), 0), expected.0, expected.1);
        }
    }

    #[test]
    fn every_safety_latch_category_has_its_own_alert() {
        let status = normal_status();
        for (kind, expected) in [
            (SafetyEventKind::Bump, ("BUMP", "")),
            (SafetyEventKind::Cliff, ("CLIFF", "")),
            (SafetyEventKind::WheelDrop, ("WHEEL", "DROP")),
            (SafetyEventKind::EStop, ("ESTOP", "")),
            (SafetyEventKind::Heartbeat, ("CTRL", "LOST")),
            (SafetyEventKind::Tilt, ("TILT", "")),
            (SafetyEventKind::Impact, ("IMPACT", "")),
            (SafetyEventKind::Charging, ("NO", "DRIVE")),
        ] {
            assert_lines(
                status.page(
                    DisplaySafety {
                        estop_latched: false,
                        safety_latch_kind: Some(kind),
                    },
                    0,
                ),
                expected.0,
                expected.1,
            );
        }
    }

    #[test]
    fn network_page_reports_startup_readiness_and_active_dhcp_leases() {
        let mut status = normal_status();
        status.wifi_state = 1;
        assert_lines(network_page(status), "pete-snyk", "192.168.4.1 START");
        status.wifi_state = WIFI_SERVICES_STARTED;
        status.network.active_leases = 2;
        assert_lines(network_page(status), "pete-snyk", "192.168.4.1 LEASE 2");
        status.wifi_state = WIFI_ERROR;
        assert_lines(network_page(status), "pete-snyk", "192.168.4.1 ERROR");
    }

    #[test]
    fn normal_status_uses_both_double_height_text_bands() {
        let dashboard = render(&normal_status().page(no_safety(), 0));
        assert!(dashboard[..WIDTH * 2].iter().any(|byte| *byte != 0));
        assert!(dashboard[WIDTH * 2..].iter().any(|byte| *byte != 0));
    }

    #[test]
    fn liveness_pixel_toggles_without_changing_the_selected_page() {
        let status = normal_status();
        let off_page = status.page(no_safety(), 0);
        let on_page = status.page(no_safety(), LIVENESS_TOGGLE_MS);
        assert_eq!(off_page.line1, on_page.line1);
        assert_eq!(off_page.line2, on_page.line2);

        let off = render(&off_page);
        let on = render(&on_page);
        let differences = off
            .iter()
            .zip(on.iter())
            .filter(|(left, right)| left != right)
            .count();
        assert_eq!(differences, 1);
        assert_eq!(off[FRAMEBUFFER_BYTES - 1] ^ on[FRAMEBUFFER_BYTES - 1], 0x80);
    }

    fn framebuffer_hash(framebuffer: &[u8; FRAMEBUFFER_BYTES]) -> u64 {
        framebuffer
            .iter()
            .fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
                (hash ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3)
            })
    }

    #[test]
    fn every_page_and_alert_matches_its_framebuffer_snapshot() {
        let status = normal_status();
        let mut boot = status;
        boot.runtime_state = RuntimeState::Booting as u8;
        let mut run = status;
        run.body_state = BodyState::Moving as u8;
        let mut stop = status;
        stop.oi_mode = 1;
        let mut warn = status;
        warn.runtime_state = RuntimeState::Error as u8;
        let mut controlled = status;
        controlled.authority_active = true;

        let network = |state| {
            page(
                "network",
                "",
                DisplayLayout::Network(NetworkStatus {
                    ssid_suffix: Some(1_337_420),
                    state,
                }),
            )
        };
        let alert = |icon| alert_page(icon);

        let pages = [
            ("dashboard_boot", health_page(boot)),
            ("dashboard_ready", health_page(status)),
            ("dashboard_run", health_page(run)),
            ("dashboard_stop", health_page(stop)),
            ("dashboard_warn", health_page(warn)),
            ("dashboard_controlled", health_page(controlled)),
            ("network_start", network(NetworkState::Starting)),
            ("network_ready", network(NetworkState::Ready)),
            ("network_lease", network(NetworkState::Lease(2))),
            ("network_error", network(NetworkState::Error)),
            (
                "battery",
                battery_page(BatteryStatus {
                    percent: 73,
                    charging: false,
                }),
            ),
            (
                "battery_charging",
                battery_page(BatteryStatus {
                    percent: 42,
                    charging: true,
                }),
            ),
            ("alert_bump", alert(AlertIcon::Bump)),
            ("alert_cliff", alert(AlertIcon::Cliff)),
            ("alert_wheel_drop", alert(AlertIcon::WheelDrop)),
            ("alert_estop", alert(AlertIcon::EStop)),
            ("alert_heartbeat", alert(AlertIcon::Heartbeat)),
            ("alert_tilt", alert(AlertIcon::Tilt)),
            ("alert_impact", alert(AlertIcon::Impact)),
            ("alert_charging", alert(AlertIcon::Charging)),
            ("alert_oi_link_lost", alert(AlertIcon::OiLinkLost)),
            ("alert_low_battery", alert(AlertIcon::LowBattery)),
            ("alert_battery_stale", alert(AlertIcon::BatteryStale)),
            ("alert_imu_offline", alert(AlertIcon::ImuOffline)),
            ("alert_wait_create", alert(AlertIcon::WaitCreate)),
            ("alert_power_off", alert(AlertIcon::PowerOff)),
            ("alert_create_no_rx", alert(AlertIcon::CreateNoRx)),
            ("alert_uart_framing", alert(AlertIcon::UartFraming)),
            ("alert_timeout", alert(AlertIcon::Timeout)),
            ("alert_invalid_packet", alert(AlertIcon::InvalidPacket)),
            ("alert_runtime_error", alert(AlertIcon::RuntimeError)),
        ];
        let expected = [
            0x7622_ed05_97b5_baf9,
            0x6742_ded5_5bfd_3ec3,
            0x7b1e_718d_b98d_10b9,
            0x24cc_85d2_5530_7bcf,
            0x5c33_5285_f482_079f,
            0x7fb9_312a_8a87_e61b,
            0xf7fe_64e2_9714_b28a,
            0xa1dd_390d_7fb4_0abc,
            0xfdab_7d1d_144f_86ad,
            0xc4b7_40fa_05d5_cd0e,
            0x945d_9d1a_02ec_af46,
            0xba9c_d6a8_f0d7_c0c1,
            0x12da_7ac2_b02f_ace2,
            0xb2e4_dee8_a08e_e256,
            0x6dfc_49f2_e1a4_c0ae,
            0xf182_9cdf_47ea_8586,
            0x3445_eed7_4e17_2226,
            0x05e4_a5b4_c9ab_35e6,
            0xc89b_9d26_35db_ea71,
            0x103e_5f07_8e33_c377,
            0x7900_5182_a418_2ed1,
            0x7742_6561_91ae_f3ec,
            0x8041_dfd5_0ce6_efec,
            0xfb71_5d25_25ab_1437,
            0x7daf_f066_7c51_80c6,
            0xc5ab_0cb8_2e95_be74,
            0x2811_ab0f_13a6_90d0,
            0x11f9_c1bd_36cf_e7bf,
            0x9554_cf17_f885_8c5a,
            0xfdc4_18ad_20c8_9eb4,
            0x2bfb_4a6d_1c28_fda2,
        ];
        let mut mismatches = 0;
        for ((name, page), expected_hash) in pages.iter().zip(expected) {
            let framebuffer = render(page);
            let actual_hash = framebuffer_hash(&framebuffer);
            if actual_hash != expected_hash {
                std::eprintln!("{name}: 0x{actual_hash:016x}");
                mismatches += 1;
            }
            assert!(
                framebuffer[..WIDTH * 3].iter().any(|byte| *byte != 0)
                    && framebuffer[WIDTH * 3..].iter().any(|byte| *byte != 0),
                "{name} must use both the upper and lower display bands"
            );
        }
        assert_eq!(mismatches, 0, "framebuffer snapshots changed");
    }
