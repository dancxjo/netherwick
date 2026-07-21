#[embassy_executor::task(pool_size = 3)]
async fn http_task(stack: Stack<'static>) -> ! {
    let mut rx_buffer = [0; 1024];
    let mut tx_buffer = [0; 2048];
    let mut request = [0; 1024];
    let mut json = [0; 4096];

    loop {
        let mut socket = TcpSocket::new(stack, &mut rx_buffer, &mut tx_buffer);
        socket.set_timeout(Some(Duration::from_secs(4)));

        if socket.accept(HTTP_PORT).await.is_err() {
            continue;
        }

        let n = match read_http_request(&mut socket, &mut request).await {
            Ok(n) => n,
            Err(_) => {
                socket.abort();
                continue;
            }
        };

        let uptime_ms = Instant::now().as_millis() as u32;
        status::mark_http_request(uptime_ms);
        let method = request_method(&request[..n]);
        let path = request_path(&request[..n]);
        let bootsel_accepted = cfg!(feature = "service-mode")
            && method == Some("POST")
            && path == Some("/command")
            && request_body(&request[..n]).is_some_and(|body| {
                json_str(body, "kind") == Some("bootsel") && json_service_authority_valid(body)
            });
        let result = match (method, path) {
            (Some("GET"), Some("/") | Some("/index.html")) => {
                write_response(&mut socket, "text/html; charset=utf-8", index_html()).await
            }
            (Some("GET"), Some(path)) if path == "/events" || path.starts_with("/events?") => {
                let since_seq = request_sse_cursor(&request[..n]);
                stream_sse(&mut socket, &mut json, since_seq).await
            }
            (Some("GET"), Some("/status.json")) => {
                let snapshot = status::snapshot(uptime_ms);
                match status::render_json(snapshot, &mut json) {
                    Ok(body) => {
                        write_response(&mut socket, "application/json", body.as_bytes()).await
                    }
                    Err(_) => write_plain_status(&mut socket, 500, "Internal Server Error").await,
                }
            }
            (Some("GET"), Some("/network.json")) => {
                let now = Instant::now().as_millis() as u32;
                match render_network_diagnostics(&mut json, now) {
                    Some(body) => {
                        write_response(&mut socket, "application/json", body.as_bytes()).await
                    }
                    None => write_plain_status(&mut socket, 500, "Internal Server Error").await,
                }
            }
            (Some("GET"), Some("/sessions.json")) => {
                let now = Instant::now().as_millis() as u32;
                match render_session_diagnostics(&mut json, now) {
                    Some(body) => {
                        write_response(&mut socket, "application/json", body.as_bytes()).await
                    }
                    None => write_plain_status(&mut socket, 500, "Internal Server Error").await,
                }
            }
            (Some("POST"), Some("/command")) => {
                match handle_command_request(&request[..n], &mut json) {
                    Ok(body) => {
                        write_response(&mut socket, "application/json", body.as_bytes()).await
                    }
                    Err(CommandParseError::Busy(command_id, reason)) => {
                        let body =
                            render_command_response(json.as_mut(), false, command_id, reason);
                        match body {
                            Some(body) => {
                                write_response(&mut socket, "application/json", body.as_bytes())
                                    .await
                            }
                            None => {
                                write_plain_status(&mut socket, 500, "Internal Server Error").await
                            }
                        }
                    }
                    Err(CommandParseError::BadRequest) => {
                        write_plain_status(&mut socket, 400, "Bad Request").await
                    }
                }
            }
            (Some("POST"), Some("/handshake")) => {
                let body = request_body(&request[..n]);
                let malformed = body.is_none_or(|body| session::parse_json(body).is_err());
                if malformed {
                    let rejection = render_handshake_reject(
                        &mut json,
                        "",
                        session::RejectReason::InvalidIdentity,
                    )
                    .unwrap_or("{\"kind\":\"reject\",\"reason_code\":\"internal_error\"}");
                    write_response_status(
                        &mut socket,
                        400,
                        "Bad Request",
                        "application/json",
                        rejection.as_bytes(),
                    )
                    .await
                } else {
                    match handle_handshake_json(
                        body.unwrap_or(""),
                        &mut json,
                        TransportKind::Http as u8,
                    ) {
                        Some(body) if body.contains("\"kind\":\"reject\"") => {
                            write_response_status(
                                &mut socket,
                                409,
                                "Conflict",
                                "application/json",
                                body.as_bytes(),
                            )
                            .await
                        }
                        Some(body) => {
                            write_response(&mut socket, "application/json", body.as_bytes()).await
                        }
                        None => {
                            let rejection = render_handshake_reject(
                                &mut json,
                                "",
                                session::RejectReason::InternalError,
                            )
                            .unwrap_or("{\"kind\":\"reject\",\"reason_code\":\"internal_error\"}");
                            write_response_status(
                                &mut socket,
                                500,
                                "Internal Server Error",
                                "application/json",
                                rejection.as_bytes(),
                            )
                            .await
                        }
                    }
                }
            }
            _ => write_plain_status(&mut socket, 404, "Not Found").await,
        };

        match result {
            Ok(true) => {
                status::mark_http_response_flushed();
                socket.close();
                if bootsel_accepted {
                    Timer::after_millis(150).await;
                    reset_to_usb_boot(0, 0);
                }
            }
            Ok(false) => {
                status::mark_http_response_error();
                socket.abort();
            }
            Err(_) => {
                status::mark_http_response_error();
                socket.abort();
            }
        }
    }
}

#[embassy_executor::task]
async fn websocket_task(stack: Stack<'static>) -> ! {
    let mut rx_buffer = [0; 1024];
    let mut tx_buffer = [0; 2048];
    let mut request = [0; 512];
    let mut payload = [0; 1024];
    let mut response = [0; 4096];

    loop {
        let mut socket = TcpSocket::new(stack, &mut rx_buffer, &mut tx_buffer);
        socket.set_timeout(Some(Duration::from_secs(30)));

        if socket.accept(WS_CONTROL_PORT).await.is_err() {
            continue;
        }

        let n = match socket.read(&mut request).await {
            Ok(n) => n,
            Err(_) => {
                socket.abort();
                continue;
            }
        };

        let path = request_path(&request[..n]);
        let Some(key) = websocket_key(&request[..n]) else {
            let _ = write_plain_status(&mut socket, 400, "Bad Request").await;
            socket.abort();
            continue;
        };

        if path != Some("/control") {
            let _ = write_plain_status(&mut socket, 404, "Not Found").await;
            socket.abort();
            continue;
        }

        let Some(accept_key) = websocket_accept_key(key, &mut response) else {
            socket.abort();
            continue;
        };

        if write_websocket_upgrade(&mut socket, accept_key)
            .await
            .is_err()
        {
            socket.abort();
            continue;
        }

        loop {
            match read_websocket_text(&mut socket, &mut payload).await {
                Ok(Some(body)) => {
                    if let Some(reply) = handle_websocket_message(body, &mut response) {
                        if write_websocket_text(&mut socket, reply.as_bytes())
                            .await
                            .is_err()
                        {
                            socket.abort();
                            break;
                        }
                    }
                }
                Ok(None) => {
                    socket.abort();
                    break;
                }
                Err(_) => {
                    socket.abort();
                    break;
                }
            }
        }
    }
}

async fn onboard_led_loop(control: &mut cyw43::Control<'static>) -> ! {
    let mut next_heartbeat_ms = 0;
    loop {
        let now_ms = Instant::now().as_millis() as u64;
        if let Some(blinks) = status::take_led_blinks() {
            blink_onboard_led(control, blinks).await;
            Timer::after_millis(600).await;
            continue;
        }

        if now_ms >= next_heartbeat_ms {
            blink_onboard_led(control, 1).await;
            next_heartbeat_ms = now_ms.saturating_add(LED_HEARTBEAT_INTERVAL_SECS * 1_000);
        }

        Timer::after_millis(100).await;
    }
}

async fn blink_onboard_led(control: &mut cyw43::Control<'static>, blinks: u8) {
    for _ in 0..blinks {
        let _ = control.gpio_set(0, true).await;
        Timer::after_millis(LED_BLINK_ON_MS).await;
        let _ = control.gpio_set(0, false).await;
        Timer::after_millis(LED_BLINK_OFF_MS).await;
    }
}

#[embassy_executor::task]
async fn udp_control_task(stack: Stack<'static>) -> ! {
    let mut rx_meta = [PacketMetadata::EMPTY; 2];
    let mut rx_buffer = [0; 512];
    let mut tx_meta = [PacketMetadata::EMPTY; 2];
    let mut tx_buffer = [0; 2048];
    let mut request = [0; 1024];
    let mut response = heapless::String::<4096>::new();

    loop {
        let mut socket = UdpSocket::new(
            stack,
            &mut rx_meta,
            &mut rx_buffer,
            &mut tx_meta,
            &mut tx_buffer,
        );
        if socket.bind(UDP_CONTROL_PORT).is_err() {
            Timer::after_secs(5).await;
            continue;
        }

        loop {
            let Ok((len, endpoint)) = socket.recv_from(&mut request).await else {
                continue;
            };
            let Ok(line) = core::str::from_utf8(&request[..len]) else {
                continue;
            };
            let Some(boot_to_usb) =
                handle_compact_control_line(line.trim(), &mut response, TransportKind::Udp as u8)
            else {
                continue;
            };
            let _ = socket.send_to(response.as_bytes(), endpoint).await;
            if boot_to_usb {
                Timer::after_millis(100).await;
                reset_to_usb_boot(0, 0);
            }
        }
    }
}

#[embassy_executor::task]
async fn dns_task(stack: Stack<'static>) -> ! {
    let mut rx_meta = [PacketMetadata::EMPTY; 2];
    let mut rx_buffer = [0; 512];
    let mut tx_meta = [PacketMetadata::EMPTY; 2];
    let mut tx_buffer = [0; 512];
    let mut request = [0; 512];
    let mut response = [0; 512];

    loop {
        let mut socket = UdpSocket::new(
            stack,
            &mut rx_meta,
            &mut rx_buffer,
            &mut tx_meta,
            &mut tx_buffer,
        );
        if socket.bind(DNS_PORT).is_err() {
            Timer::after_secs(5).await;
            continue;
        }

        loop {
            let Ok((len, endpoint)) = socket.recv_from(&mut request).await else {
                continue;
            };
            let Some(reply) = build_dns_reply(&request[..len], &mut response) else {
                continue;
            };
            let _ = socket.send_to(reply, endpoint).await;
        }
    }
}

#[embassy_executor::task]
async fn mdns_task(stack: Stack<'static>) -> ! {
    let mut rx_meta = [PacketMetadata::EMPTY; 2];
    let mut rx_buffer = [0; 256];
    let mut tx_meta = [PacketMetadata::EMPTY; 2];
    let mut tx_buffer = [0; 768];
    let mut packet = [0; 768];
    let endpoint = IpEndpoint::new(IpAddress::Ipv4(Ipv4Address::new(224, 0, 0, 251)), MDNS_PORT);

    loop {
        let mut socket = UdpSocket::new(
            stack,
            &mut rx_meta,
            &mut rx_buffer,
            &mut tx_meta,
            &mut tx_buffer,
        );
        if socket.bind(MDNS_PORT).is_ok() {
            loop {
                let len = build_mdns_announcement(&mut packet);
                let _ = socket.send_to(&packet[..len], endpoint).await;
                Timer::after_secs(5).await;
            }
        }
        Timer::after_secs(5).await;
    }
}

#[embassy_executor::task]
async fn dhcp_task(stack: Stack<'static>) -> ! {
    let mut rx_meta = [PacketMetadata::EMPTY; 2];
    let mut rx_buffer = [0; 1024];
    let mut tx_meta = [PacketMetadata::EMPTY; 2];
    let mut tx_buffer = [0; 1024];
    let mut request = [0; 576];
    let mut response = [0; 576];
    let mut leases = DhcpLeaseState::new();
    let endpoint = IpEndpoint::new(
        IpAddress::Ipv4(Ipv4Address::new(255, 255, 255, 255)),
        DHCP_CLIENT_PORT,
    );

    loop {
        let mut socket = UdpSocket::new(
            stack,
            &mut rx_meta,
            &mut rx_buffer,
            &mut tx_meta,
            &mut tx_buffer,
        );

        if socket.bind(DHCP_SERVER_PORT).is_err() {
            Timer::after_secs(5).await;
            continue;
        }

        loop {
            let Ok((len, _meta)) = socket.recv_from(&mut request).await else {
                continue;
            };

            let Some(dhcp_request) = DhcpRequest::parse(&request[..len]) else {
                continue;
            };
            let Some(grant) = leases.grant(dhcp_request, Instant::now().as_millis() as u64) else {
                continue;
            };
            let client = dhcp_request.client();
            network_registry::record_lease(
                client.lease_identity(),
                grant.lease_ip(),
                (Instant::now().as_millis() as u32).wrapping_add(DHCP_LEASE_SECONDS * 1_000),
            );
            let Some(reply) = build_dhcp_reply(grant, &request[..len], &mut response) else {
                continue;
            };
            status::mark_dhcp_grant();
            let _ = socket.send_to(reply, endpoint).await;
        }
    }
}
