    use super::*;

    #[test]
    fn get_events_requires_a_numeric_json_cursor() {
        assert!(matches!(
            parse_command(
                103,
                r#"{"command_id":103,"kind":"get_events","since_seq":491}"#
            ),
            Some(BrainstemCommand::GetEvents { since_seq: 491 })
        ));
        assert!(parse_command(
            103,
            r#"{"command_id":103,"kind":"get_events","since_seq":NEXT_SEQ}"#
        )
        .is_none());
        assert!(parse_command(103, r#"{"command_id":103,"kind":"get_events"}"#).is_none());
        assert!(parse_command(
            103,
            r#"{"command_id":103,"kind":"get_events","since_seq":491garbage}"#
        )
        .is_none());
    }

    #[test]
    fn silent_mode_parses_on_http_and_compact_uart() {
        assert!(matches!(
            parse_command(
                104,
                r#"{"command_id":104,"kind":"set_silent","silent":true,"seq":104}"#
            ),
            Some(BrainstemCommand::SetAudioSilent {
                silent: true,
                seq: 104
            })
        ));
        assert!(matches!(
            parse_forebrain_uart_command("SET_SILENT 105 false"),
            Ok((
                105,
                BrainstemCommand::SetAudioSilent {
                    silent: false,
                    seq: 105
                }
            ))
        ));
    }

    #[test]
    fn compact_get_events_requires_a_numeric_cursor() {
        assert!(matches!(
            parse_forebrain_uart_command("GET_EVENTS 103 491"),
            Ok((103, BrainstemCommand::GetEvents { since_seq: 491 }))
        ));
        assert_eq!(
            parse_forebrain_uart_command("GET_EVENTS 103 NEXT_SEQ"),
            Err(103)
        );
        assert_eq!(parse_forebrain_uart_command("GET_EVENTS 103"), Err(103));
    }

    #[test]
    fn retired_compact_commands_ignore_legacy_arguments_and_reach_typed_rejection() {
        assert!(matches!(
            parse_forebrain_uart_command("DRIVE_FOR 104 100 50 1000"),
            Ok((104, BrainstemCommand::Unsupported { seq: 104 }))
        ));
        assert!(matches!(
            parse_forebrain_uart_command("SET_SAFETY_POLICY 105 none none false"),
            Ok((105, BrainstemCommand::Unsupported { seq: 105 }))
        ));
    }
