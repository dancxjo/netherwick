async fn write_response(
    socket: &mut TcpSocket<'_>,
    content_type: &str,
    body: &[u8],
) -> Result<bool, embassy_net::tcp::Error> {
    let mut header = heapless::String::<192>::new();
    let _ = write!(
        header,
        "HTTP/1.1 200 OK\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        content_type,
        body.len()
    );
    socket.write_all(header.as_bytes()).await?;
    socket.write_all(body).await?;
    flush_tcp_with_timeout(socket).await
}

async fn stream_sse(
    socket: &mut TcpSocket<'_>,
    json: &mut [u8],
    mut since_seq: u32,
) -> Result<bool, embassy_net::tcp::Error> {
    socket
        .write_all(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\n\r\nretry: 1000\r\n\r\n",
        )
        .await?;
    if !flush_tcp_with_timeout(socket).await? {
        return Ok(false);
    }

    let mut next_status_ms = 0;
    loop {
        let now_ms = Instant::now().as_millis() as u64;
        if now_ms >= next_status_ms {
            let snapshot = status::snapshot(now_ms as u32);
            let body = match status::render_json(snapshot, json) {
                Ok(body) => body,
                Err(_) => return Ok(false),
            };
            match write_sse_event(socket, "status", None, body).await {
                Ok(true) => {}
                Ok(false) | Err(_) => return Ok(true),
            }
            next_status_ms = now_ms.saturating_add(SSE_STATUS_INTERVAL_MS);
        }

        let event_next_seq = status::event_next_seq();
        if event_next_seq != since_seq.saturating_add(1) {
            let body = match status::render_events_json(since_seq, json) {
                Some(body) => body,
                None => return Ok(false),
            };
            let last_seq = json_u32(body, "next_seq")
                .unwrap_or(event_next_seq)
                .saturating_sub(1);
            match write_sse_event(socket, "events", Some(last_seq), body).await {
                Ok(true) => since_seq = last_seq,
                Ok(false) | Err(_) => return Ok(true),
            }
        }

        Timer::after_millis(SSE_EVENT_CHECK_INTERVAL_MS).await;
    }
}

async fn write_sse_event(
    socket: &mut TcpSocket<'_>,
    event: &str,
    id: Option<u32>,
    body: &str,
) -> Result<bool, embassy_net::tcp::Error> {
    let mut prefix = heapless::String::<64>::new();
    let _ = write!(prefix, "event: {event}\r\n");
    if let Some(id) = id {
        let _ = write!(prefix, "id: {id}\r\n");
    }
    let _ = prefix.push_str("data: ");
    socket.write_all(prefix.as_bytes()).await?;
    socket.write_all(body.trim_end().as_bytes()).await?;
    socket.write_all(b"\r\n\r\n").await?;
    flush_tcp_with_timeout(socket).await
}

async fn read_http_request(
    socket: &mut TcpSocket<'_>,
    buffer: &mut [u8],
) -> Result<usize, embassy_net::tcp::Error> {
    let mut used = 0;
    loop {
        if used == buffer.len() {
            return Ok(used);
        }
        let read = socket.read(&mut buffer[used..]).await?;
        if read == 0 {
            return Ok(used);
        }
        used += read;
        let Some(header_end) = buffer[..used]
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .map(|index| index + 4)
        else {
            continue;
        };
        let header = core::str::from_utf8(&buffer[..header_end]).unwrap_or("");
        let content_length = header
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("Content-Length")
                    .then(|| value.trim().parse::<usize>().ok())
                    .flatten()
            })
            .unwrap_or(0);
        if used >= header_end.saturating_add(content_length) {
            return Ok(used);
        }
    }
}

async fn write_response_status(
    socket: &mut TcpSocket<'_>,
    code: u16,
    text: &str,
    content_type: &str,
    body: &[u8],
) -> Result<bool, embassy_net::tcp::Error> {
    let mut header = heapless::String::<192>::new();
    let _ = write!(
        header,
        "HTTP/1.1 {code} {text}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    socket.write_all(header.as_bytes()).await?;
    socket.write_all(body).await?;
    flush_tcp_with_timeout(socket).await
}

async fn write_plain_status(
    socket: &mut TcpSocket<'_>,
    code: u16,
    text: &str,
) -> Result<bool, embassy_net::tcp::Error> {
    let mut header = heapless::String::<160>::new();
    let _ = write!(
        header,
        "HTTP/1.1 {} {}\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        code,
        text,
        text.len(),
        text
    );
    socket.write_all(header.as_bytes()).await?;
    flush_tcp_with_timeout(socket).await
}

async fn flush_tcp_with_timeout(
    socket: &mut TcpSocket<'_>,
) -> Result<bool, embassy_net::tcp::Error> {
    match select(socket.flush(), Timer::after_millis(HTTP_FLUSH_TIMEOUT_MS)).await {
        Either::First(result) => result.map(|()| true),
        Either::Second(()) => Ok(false),
    }
}

fn request_path(request: &[u8]) -> Option<&str> {
    let line_end = request
        .windows(2)
        .position(|w| w == b"\r\n")
        .unwrap_or(request.len());
    let line = core::str::from_utf8(&request[..line_end]).ok()?;
    let mut parts = line.split(' ');
    let _method = parts.next()?;
    parts.next()
}

fn request_method(request: &[u8]) -> Option<&str> {
    let line_end = request
        .windows(2)
        .position(|w| w == b"\r\n")
        .unwrap_or(request.len());
    let line = core::str::from_utf8(&request[..line_end]).ok()?;
    line.split(' ').next()
}

fn request_sse_cursor(request: &[u8]) -> u32 {
    let requested = request_header(request, "Last-Event-ID")
        .and_then(|value| value.parse().ok())
        .or_else(|| {
            let query = request_path(request)?.split_once('?')?.1;
            query.split('&').find_map(|field| {
                let (name, value) = field.split_once('=')?;
                (name == "since").then(|| value.parse().ok()).flatten()
            })
        });
    let next_seq = status::event_next_seq();
    match requested {
        Some(cursor) if cursor < next_seq => cursor,
        Some(_) => 0,
        None => next_seq.saturating_sub(1),
    }
}

fn request_header<'a>(request: &'a [u8], wanted: &str) -> Option<&'a str> {
    let request = core::str::from_utf8(request).ok()?;
    request.lines().skip(1).find_map(|line| {
        let (name, value) = line.split_once(':')?;
        name.eq_ignore_ascii_case(wanted).then(|| value.trim())
    })
}

fn index_html() -> &'static [u8] {
    // Embedded browser cockpit mapping to the host-side pete-cockpit contract:
    //
    // UI action                    JSON kind             CockpitRequest                 Capability
    // joystick / drive pad          cmd_vel               CmdVel                         cmd_vel
    // active motion heartbeat       heartbeat_stop        HeartbeatStop                  heartbeat_stop
    // STOP                          stop                  Stop                           stop
    // E-STOP                        estop                 EStop                          estop
    // Clear E-Stop                  clear_estop           ClearEStop                     clear_estop
    // Dock                          dock                  Dock                           dock
    // Ping                          ping                  Ping                           ping
    // Music Define / Play           song_define/play      SongDefine / SongPlay          song_define/song_play
    // Silent mode                   set_silent            SetAudioSilent                 set_silent
    // Refresh                       reconnect /events SSE (no command)
    // BOOTSEL                       bootsel               Bootsel                        service/debug only
    br#"<!doctype html>
<html>
<head>
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pete Brainstem</title>
<style>
:root{font-family:system-ui,-apple-system,Segoe UI,sans-serif;color:#1f2728;background:#eef1ed;accent-color:#1d7580}
*{box-sizing:border-box}body{margin:0}.wrap{max-width:1280px;margin:auto;padding:14px}
header{display:grid;grid-template-columns:minmax(210px,1fr) auto;gap:12px;align-items:start;margin-bottom:12px}
h1{font-size:23px;line-height:1.05;margin:0;color:#121817}h2{font-size:12px;margin:0;color:#596361;text-transform:uppercase;font-weight:850}
.sub{font-size:13px;color:#697370;margin-top:5px}.top{display:flex;gap:7px;flex-wrap:wrap;justify-content:flex-end;align-items:center}
.pill{font-size:12px;border:1px solid #c5cdc8;border-radius:999px;padding:5px 9px;background:#fff;color:#303936;white-space:nowrap}
.pill.ok{border-color:#5ca77a;background:#edf8f1}.pill.warn{border-color:#d4aa40;background:#fff7dc}.pill.bad{border-color:#cf6868;background:#fff0f0}
.check{display:inline-flex;align-items:center;gap:5px;font-size:12px;font-weight:750;color:#44504a;white-space:nowrap}.check input{width:16px;min-height:16px;margin:0}.lock{font-size:12px;border:1px solid #c5cdc8;border-radius:999px;padding:5px 9px;background:#fff;color:#68736c;font-weight:850;white-space:nowrap}.lock.ok{border-color:#5ca77a;background:#edf8f1;color:#287142}.lock.warn{border-color:#d4aa40;background:#fff7dc;color:#6d5510}
.layout{display:grid;grid-template-columns:minmax(340px,.95fr) minmax(0,1.45fr);gap:10px;align-items:start}
.side{display:grid;gap:10px}.station{background:#fff;border:1px solid #d7ded9;border-radius:8px;padding:11px;box-shadow:0 1px 2px #17241c10;display:grid;gap:10px}
.station-head{display:flex;align-items:center;justify-content:space-between;gap:8px}.station-head .pill{padding:4px 8px}.motion{position:sticky;top:10px}
.zone{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(180px,.85fr);gap:10px;align-items:start}.zone.slim{grid-template-columns:minmax(0,1fr) minmax(150px,.55fr)}
.controls{display:grid;gap:8px}.joy{min-height:326px;display:grid;place-items:center;touch-action:none;user-select:none;background:#f7f9f7;border:1px solid #e1e7e3;border-radius:8px}
.base{width:min(68vw,296px);height:min(68vw,296px);border-radius:50%;background:#e5ebe7;border:2px solid #c3ccc6;position:relative;box-shadow:inset 0 0 0 28px #f0f4f1}
.base:before,.base:after{content:"";position:absolute;background:#c8d1cb}.base:before{width:2px;height:82%;left:50%;top:9%}.base:after{height:2px;width:82%;left:9%;top:50%}
.nub{width:84px;height:84px;border-radius:50%;background:#1d7580;position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);box-shadow:0 8px 18px #13251c33;border:4px solid #fbfdfb}
.row{display:flex;gap:8px;flex-wrap:wrap}.row>*{flex:1 1 auto}.split{display:grid;grid-template-columns:1fr 1fr;gap:8px}
button{min-height:40px;border:1px solid #b9c2bd;border-radius:7px;background:#fff;color:#202722;font-weight:750;font-size:14px;letter-spacing:0;cursor:pointer}
button:active,.active{transform:translateY(1px);background:#eef2ef}button:disabled{opacity:.48;cursor:not-allowed}.primary{background:#dceee6;border-color:#8eb99f}.stop{background:#202522;color:#fff;border-color:#202522}.danger{background:#9d2830;color:#fff;border-color:#842029}.warnbtn{background:#fff3d6;border-color:#d8b24a}.blue{background:#e7f0fb;border-color:#9bbbe0}
.pad{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px}.pad button{min-height:48px}.pad .center{grid-column:2}
.seg{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:7px}.seg.three{grid-template-columns:repeat(3,minmax(0,1fr))}.seg button{min-height:38px;font-size:12px}
label{font-size:12px;color:#5b655f;font-weight:750}.slider,.field{display:grid;gap:6px}.slider input{width:100%}input,select{width:100%;min-height:40px;border:1px solid #cbd3ce;border-radius:7px;padding:8px;font:inherit;background:#fff}
.readout{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;font-size:13px}.readout.compact{grid-template-columns:1fr}.tile{background:#f6f8f6;border:1px solid #e1e6e2;border-radius:7px;padding:8px;min-height:50px}
.tile b{display:block;color:#4e5852;font-size:11px;text-transform:uppercase;margin-bottom:3px}.tile span,.tile div{overflow-wrap:anywhere}.wide{grid-column:1/-1}.muted{color:#68736c}.badtext{color:#a1262f}.oktext{color:#287142}
.imu{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px}.imu .tile{min-height:58px}.bar{height:7px;border-radius:999px;background:#dfe6e2;overflow:hidden;margin-top:6px}.bar i{display:block;height:100%;width:0;background:#1d7580}.bar.warn i{background:#d49832}.bar.bad i{background:#b12c37}
.log{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;line-height:1.45;max-height:132px;overflow:auto;white-space:pre-wrap}
@media(max-width:980px){.layout{grid-template-columns:1fr}.motion{position:static}.zone,.zone.slim{grid-template-columns:1fr}.imu{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:560px){.wrap{padding:10px}header{grid-template-columns:1fr}.top{justify-content:flex-start}.readout,.split,.imu{grid-template-columns:1fr}.seg,.seg.three{grid-template-columns:repeat(2,minmax(0,1fr))}}
</style>
</head>
<body>
<div class="wrap">
<header><div><h1>Pete Brainstem</h1><div class="sub" id="headline">Waiting for status</div></div><div class="top"><span id="session" class="pill">no session</span><button id="sessionnew">New session</button><button id="controllease">Request control</button><span id="controlstate" class="lock">unlocked</span><label class="check"><input id="controlrefresh" type="checkbox">Keep control</label><span id="net" class="pill">connecting</span><span id="mode" class="pill">mode unknown</span><span id="safety" class="pill">safety unknown</span></div></header>
<div class="layout">
<section class="station motion">
<div class="station-head"><h2>Drive</h2><span id="cmd" class="pill">command unknown</span></div>
<div class="joy"><div id="base" class="base"><div id="nub" class="nub"></div></div></div>
<div class="split"><div class="slider"><label for="speed">Speed <span id="speedv">120</span> mm/s</label><input id="speed" type="range" min="40" max="260" value="120"></div><div class="slider"><label for="turn">Turn <span id="turnv">1200</span> mrad/s</label><input id="turn" type="range" min="300" max="2000" value="1200"></div></div>
<div class="pad">
<button class="primary center" data-drive="fwd">FWD</button>
<button data-drive="left">LEFT</button><button class="stop" id="padstop">STOP</button><button data-drive="right">RIGHT</button>
<button data-drive="back" class="center">BACK</button>
<button data-drive="spinl">SPIN L</button><button data-drive="slow">SLOW</button><button data-drive="spinr">SPIN R</button>
</div>
<div class="row"><button class="stop" id="stop">STOP</button></div>
</section>
<div class="side">
<section class="station">
<div class="station-head"><h2>Safety and Reflexes</h2><span class="pill">reflex guard</span></div>
<div class="zone slim">
<div class="readout">
<div class="tile wide"><b>Safety</b><span id="safetyread" class="muted">...</span></div>
<div class="tile"><b>Last error</b><span id="err" class="muted">...</span></div>
<div class="tile"><b>Events</b><span id="events" class="muted">...</span></div>
</div>
<div class="controls">
<button class="warnbtn" id="careful">CAREFUL - I HAVE THE BODY (5 s)</button>
<div class="seg three"><button class="danger" id="estop">E-STOP</button><button id="clear">Clear E-Stop</button><button id="clearcharge">Clear Charge</button></div>
<div class="seg three"><button id="clearbump">Clear Bump</button><button id="clearwheel">Clear Wheel</button><button id="clearcliff">Clear Cliff</button></div>
<div class="seg"><button id="cleartilt">Clear Tilt</button><button id="clearimpact">Clear Impact</button></div>
<button class="blue" data-action="heartbeat">Heartbeat</button>
</div>
</div>
</section>
<section class="station">
<div class="station-head"><h2>Create Body</h2><span id="create" class="pill">create unknown</span></div>
<div class="zone">
<div class="readout">
<div class="tile"><b>Battery</b><span id="battery" class="muted">...</span></div>
<div class="tile"><b>Odometry</b><span id="odom" class="muted">...</span></div>
<div class="tile wide"><b>Sensors</b><span id="sensors" class="muted">...</span></div>
</div>
<div class="controls">
<div class="seg three"><button id="undock">Undock</button><button id="dock">Dock</button><button id="ping">Ping</button></div>
<div class="seg"><button id="stream">Stream Sensors</button><button id="createon" class="blue">Create On</button></div>
<div class="seg"><button id="createoi">Start OI</button></div>
<div class="seg"><button id="createoff" class="warnbtn">Create Off</button><button id="createrestart" class="warnbtn">Restart Create</button></div>
</div>
</div>
</section>
<section class="station">
<div class="station-head"><h2>IMU</h2><span class="pill">orientation</span></div>
<div class="zone">
<div class="imu">
<div class="tile"><b>IMU health</b><span id="imuhealth" class="muted">...</span></div>
<div class="tile"><b>Yaw</b><span id="imuyaw" class="muted">...</span></div>
<div class="tile"><b>Accel</b><span id="imuaccel" class="muted">...</span><div id="imuaccelbar" class="bar"><i></i></div></div>
<div class="tile"><b>Tilt</b><span id="imutilt" class="muted">...</span><div id="imutiltbar" class="bar"><i></i></div></div>
<div class="tile"><b>Angular rate</b><span id="imurates" class="muted">...</span></div>
<div class="tile"><b>Roughness</b><span id="imurough" class="muted">...</span><div id="imuroughbar" class="bar"><i></i></div></div>
<div class="tile"><b>Impact</b><span id="imuimpact" class="muted">...</span><div id="imuimpactbar" class="bar"><i></i></div></div>
<div class="tile"><b>Motion</b><span id="imumotion" class="muted">...</span></div>
</div>
<div class="controls">
<button id="imuzero" class="primary">Zero IMU</button>
<button id="imuclear">Clear IMU</button>
</div>
</div>
</section>
<section class="station">
<div class="station-head"><h2>Session and Link</h2><button id="refresh">Refresh</button></div>
<div class="zone slim">
<div class="readout">
<div class="tile"><b>Runtime</b><span id="runtime" class="muted">...</span></div>
<div class="tile"><b>Uptime</b><span id="uptime" class="muted">...</span></div>
<div class="tile"><b>UART</b><span id="uart" class="muted">...</span></div>
<div class="tile"><b>Forebrain</b><span id="forebrain" class="muted">...</span></div>
<div class="tile"><b>Web</b><span id="web" class="muted">...</span></div>
<div class="tile"><b>Firmware</b><span id="firmware" class="muted">...</span></div>
</div>
<div class="controls"><div class="seg"><button id="mbreset" class="danger">Reset Motherbrain</button><button id="bootsel" class="danger">BOOTSEL</button></div></div>
</div>
</section>
<section class="station">
<div class="station-head"><h2>Music</h2><label class="check"><input id="silent" type="checkbox">Silent mode</label><span id="music" class="pill">music unknown</span></div>
<div class="zone slim">
<div class="split"><div class="field"><label for="songid">Slot</label><input id="songid" inputmode="numeric" value="0"></div><div class="field"><label for="tones">Tones</label><input id="tones" value="72:8,76:8,79:16"></div></div>
<div class="controls"><div class="seg three"><button id="songdef" class="primary">Define</button><button id="songplay">Play</button><button id="song">Chirp</button></div></div>
</div>
</section>
<section class="station">
<div class="station-head"><h2>Activity</h2></div>
<div id="log" class="log muted">No commands yet</div>
</section>
</div>
</div>
</div>
<script>
let id=1,active=false,timer=0,controlRefreshTimer=0,controlRefreshGeneration=0,controlLeaseExpiresAt=0,browserSessionPromise=null,last={x:0,y:0},ws=null,wsOpen=false,sse=null,sseOpen=false,driveKind='',lastDriveAt=0,lastHeartbeatAt=0,eventCursor=0,caps=null,lastStatus=null,sessionId='',controlLeaseId='',serviceLeaseId='',sensorStreamRequested=false;
const $=x=>document.getElementById(x),base=$('base'),nub=$('nub'),net=$('net'),log=$('log');
const seqKinds=new Set(['cmd_vel','drive_direct','drive_arc','clear_safety_latch','careful_mode','escape_motion','heartbeat_stop','request_sensors','stream_sensors','clear_motion_queue','define_chirp','play_feedback','power_state','create_power_on','create_power_off','calibrate_turn','orientation_probe','reset_odometry','zero_imu_orientation','clear_imu_orientation','song_define']);
const actionVerb={heartbeat:'heartbeat_stop'};
function title(s){return (s||'unknown').replaceAll('_',' ')}
function errorName(code){return ({1:'Create did not respond',2:'Create UART framing failure',3:'Create command timeout',4:'invalid Create sensor packet'})[code]||('brainstem error '+code)}
function pill(el,text,state){el.textContent=text;el.className='pill '+(state||'')}
function addLog(text){let t=new Date().toLocaleTimeString();log.textContent=(t+'  '+text+'\n'+(log.textContent==='No commands yet'?'':log.textContent)).slice(0,900)}
function hasVerb(v){return !!(caps&&caps.verbs&&caps.verbs.indexOf(v)>=0)}
function hasOutput(v){return !!(caps&&caps.outputs&&caps.outputs.indexOf(v)>=0)}
function setEnabled(id,on){let e=$(id);if(e)e.disabled=!on}
function setEnabledAll(selector,on){document.querySelectorAll(selector).forEach(e=>e.disabled=!on)}
function chargeActive(cs){return cs.charging_indicator==='on'||(cs.charging_state>=1&&cs.charging_state<=3)}
function homeBaseContact(cs){return !!((cs.charging_sources||0)&2)}
function statusBlocksMotion(){let s=lastStatus||{},cs=s.create_sensors||{},imu=s.imu||{},dock=homeBaseContact(cs),imuDanger=imu.health==='fault'||(imu.health==='ok'&&((imu.tilt_magnitude_mrad||0)>=650||(imu.impact_score_mm_s2||0)>=18000)),safety=s.estop_latched||(!s.careful_mode_active&&(s.safety_tripped||s.motion_interlock_latched||cs.wheel_drop||(!dock&&(cs.cliff_left||cs.cliff_front_left||cs.cliff_front_right||cs.cliff_right))||imuDanger));return !!safety}
function canSession(verb){return hasVerb(verb)&&!!sessionId}
function canControl(verb){return hasVerb(verb)&&!!sessionId&&!!controlLeaseId}
function canMotion(verb){return canControl(verb)&&!statusBlocksMotion()}
function canService(verb){return hasVerb(verb)&&!!sessionId&&!!serviceLeaseId}
function ensureSensorStream(){if(sensorStreamRequested||!sessionId||!hasVerb('stream_sensors'))return;sensorStreamRequested=true;sendCockpit({kind:'stream_sensors',enabled:true,packet_id:0,period_ms:250},false).then(j=>{if(j&&j.accepted===false)sensorStreamRequested=false})}
function token(prefix){let a=new Uint32Array(2);crypto.getRandomValues(a);return prefix+'-'+a[0].toString(16)+'-'+a[1].toString(16)}
function controlLock(text,state){let e=$('controlstate');e.textContent=text;e.className='lock '+(state||'')}
function refreshControlLock(){controlLock(controlLeaseId?'locked':'unlocked',controlLeaseId?'ok':($('controlrefresh').checked?'warn':''))}
function stopControlRefresh(){clearTimeout(controlRefreshTimer);controlRefreshTimer=0;controlRefreshGeneration++}
function scheduleControlRefresh(delayMs){clearTimeout(controlRefreshTimer);controlRefreshTimer=0;if(!$('controlrefresh').checked)return;let generation=controlRefreshGeneration;controlRefreshTimer=setTimeout(()=>{if(generation!==controlRefreshGeneration||!$('controlrefresh').checked)return;if(sessionId)acquireBrowserControl();else establishBrowserSession().catch(()=>scheduleControlRefresh(5000))},Math.max(250,delayMs))}
function retryControlRefresh(){if(!$('controlrefresh').checked)return;let remaining=controlLeaseExpiresAt-Date.now();scheduleControlRefresh(remaining>1000?Math.min(5000,remaining-1000):5000)}
function establishBrowserSession(){if(browserSessionPromise)return browserSessionPromise;stopControlRefresh();controlLeaseId='';controlLeaseExpiresAt=0;serviceLeaseId='';sensorStreamRequested=false;applyCaps();let boot=sessionStorage.getItem('pete-browser-boot');if(!boot){boot=token('browserboot');sessionStorage.setItem('pete-browser-boot',boot)}let nonce=token('hello'),hello={role:'operator',session_purpose:'control',device_id:token('browser'),boot_id:boot,handshake_nonce:nonce,protocol_major:1,protocol_minor_min:0,protocol_minor_max:0,supported_features:['session_ids','event_cursor','heartbeat','transport_failover'],required_features:['session_ids'],preferred_heartbeat_ms:500};pill($('session'),'handshaking','warn');let pending=fetch('/handshake',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(hello)}).then(r=>r.json()).then(j=>{if(j.kind!=='welcome'||j.echoed_handshake_nonce!==nonce)throw new Error(j.reason_code||'invalid welcome');sessionId=j.session_id;eventCursor=Math.max(0,(j.current_event_next_seq||1)-1);pill($('session'),'session '+sessionId.slice(-8),'ok');pill(net,'session HTTP','ok');addLog('session opened '+sessionId);applyCaps();requestCaps();connectSse();if($('controlrefresh').checked)acquireBrowserControl();return j}).catch(e=>{sessionId='';controlLeaseId='';controlLeaseExpiresAt=0;serviceLeaseId='';sensorStreamRequested=false;applyCaps();pill($('session'),'session failed','bad');addLog('handshake failed '+e.message);if($('controlrefresh').checked)scheduleControlRefresh(5000);throw e});browserSessionPromise=pending;pending.then(()=>{if(browserSessionPromise===pending)browserSessionPromise=null},()=>{if(browserSessionPromise===pending)browserSessionPromise=null});return pending}
function acquireBrowserControl(){if(!sessionId){if($('controlrefresh').checked)return establishBrowserSession();addLog('open a session first');return Promise.resolve(null)}let requestedSessionId=sessionId,body={kind:'acquire_control_lease',command_id:id++,session_id:requestedSessionId,authority:'operator_debug',ttl_ms:60000};controlLock('refreshing','warn');return fetch('/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}).then(r=>r.json()).then(j=>{if(sessionId!==requestedSessionId)return j;if(j.accepted&&j.type==='control_lease_granted'){controlLeaseId=j.lease_id;controlLeaseExpiresAt=Date.now()+(j.ttl_ms||60000);serviceLeaseId='';pill($('session'),'operator control','ok');addLog('control lease '+j.lease_id+' ('+(j.ttl_ms||60000)+' ms)');applyCaps();if($('controlrefresh').checked)scheduleControlRefresh(Math.max(1000,(j.ttl_ms||60000)-15000))}else{handleReply(j);applyCaps();retryControlRefresh()}refreshControlLock();return j}).catch(e=>{if(sessionId===requestedSessionId){refreshControlLock();addLog('control request failed '+e.message);retryControlRefresh()}return null})}
function syncControlRefresh(){stopControlRefresh();refreshControlLock();if($('controlrefresh').checked){if(sessionId)acquireBrowserControl();else establishBrowserSession().catch(()=>{})}}
function acquireService(scope){if(!sessionId){addLog('open a session first');return Promise.resolve(null)}let body={kind:'acquire_service_lease',command_id:id++,session_id:sessionId,scope,ttl_ms:5000};return fetch('/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}).then(r=>r.json()).then(j=>{if(j.accepted&&j.type==='service_lease_granted'){serviceLeaseId=j.lease_id;controlLeaseId='';pill($('session'),'service '+scope,'warn');addLog('service lease '+scope);applyCaps()}else{serviceLeaseId='';handleReply(j);applyCaps()}refreshControlLock();return j}).catch(e=>{serviceLeaseId='';addLog('service request failed '+e.message);applyCaps();return null})}
function serviceCommand(scope,cmd){return acquireService(scope).then(j=>{if(!(j&&j.accepted))return j;return sendCockpit(cmd).then(r=>{serviceLeaseId='';applyCaps();if($('controlrefresh').checked&&scope!=='bootsel'&&scope!=='reset_motherbrain')acquireBrowserControl();return r})})}
function connectWs(){try{ws=new WebSocket('ws://'+location.hostname+':81/control');ws.onopen=()=>{wsOpen=true;pill(net,'control ws','ok');requestCaps()};ws.onclose=()=>{wsOpen=false;pill(net,'reconnecting','warn');setTimeout(connectWs,1000)};ws.onerror=()=>{wsOpen=false;pill(net,'ws error','warn')};ws.onmessage=e=>{try{handleReply(JSON.parse(e.data))}catch(_){}}}catch(_){wsOpen=false}}
function connectSse(){if(sse)sse.close();sseOpen=false;sse=new EventSource('/events?since='+eventCursor);sse.onopen=()=>{sseOpen=true;pill(net,'telemetry sse','ok')};sse.addEventListener('status',e=>{try{showStatus(JSON.parse(e.data))}catch(_){}});sse.addEventListener('events',e=>{try{handleEvents(JSON.parse(e.data))}catch(_){}});sse.onerror=()=>{sseOpen=false;pill(net,'sse reconnecting','warn')}}
function handleReply(j){if(j.type==='status'){showStatus(j);return}if(j.type==='events'){handleEvents(j);return}if(j.verbs){caps=j;applyCaps();pill(net,'capabilities','ok');ensureSensorStream();return}let ok=j.accepted!==false,reason=j.message||j.reason||'';if(!ok&&(reason==='invalid_control_lease'||reason==='control_lease_required')){controlLeaseId='';pill($('session'),'session; control expired','warn');addLog('control authority expired; press Request control')}if(!ok&&(reason==='invalid_service_lease'||reason==='service_authorization_required'||reason==='service_operation_disabled')){serviceLeaseId=''}if(!ok&&(reason==='invalid_session'||reason==='session_required')){sessionId='';controlLeaseId='';serviceLeaseId='';sensorStreamRequested=false;pill($('session'),'session expired','bad');addLog('session expired; press New session')}applyCaps();refreshControlLock();pill(net,ok?'accepted':'rejected',ok?'ok':'warn');if(!ok)addLog('rejected '+(reason||j.command_id||''));else if(j.message)addLog(j.message+' '+(j.command_id||''))}
function sendCockpit(o,ack){let cid=id++;o.command_id=cid;if(sessionId)o.session_id=sessionId;if(controlLeaseId)o.lease_id=controlLeaseId;if(serviceLeaseId)o.service_lease_id=serviceLeaseId;if(seqKinds.has(o.kind)&&o.seq===undefined)o.seq=cid;let body=JSON.stringify(o),name=o.kind==='cmd_vel'?'drive':o.kind;return fetch('/command',{method:'POST',headers:{'Content-Type':'application/json'},body}).then(r=>r.json()).then(j=>{handleReply(j);return j}).catch(_=>{pill(net,'offline','bad');addLog('offline '+name)})}
function requestStatus(){return sendCockpit({kind:'status'},false)}
function requestCaps(){return sendCockpit({kind:'get_capabilities'},false).then(j=>{if(j&&j.verbs){caps=j;applyCaps();ensureSensorStream()}})}
function applyCaps(){let drive=canMotion('cmd_vel'),svc=!!sessionId,canClearLatch=canControl('clear_safety_latch'),docked=!!(lastStatus&&lastStatus.create_sensors&&homeBaseContact(lastStatus.create_sensors));setEnabled('controllease',!!sessionId);setEnabled('stop',hasVerb('stop'));setEnabled('padstop',hasVerb('stop'));setEnabled('estop',hasVerb('estop'));setEnabled('clear',canSession('clear_estop'));setEnabled('careful',canControl('careful_mode'));setEnabled('clearcharge',canClearLatch&&!(lastStatus&&lastStatus.create_sensors&&chargeActive(lastStatus.create_sensors)));['clearbump','clearwheel','clearcliff','cleartilt','clearimpact'].forEach(id=>setEnabled(id,canClearLatch));setEnabled('stream',canSession('stream_sensors'));setEnabled('imuzero',canControl('zero_imu_orientation'));setEnabled('imuclear',canControl('clear_imu_orientation'));setEnabled('createrestart',svc&&hasVerb('restart_create'));setEnabled('mbreset',svc&&hasVerb('reset_motherbrain'));setEnabled('bootsel',svc&&hasVerb('bootsel'));setEnabled('createon',canControl('create_power_on'));setEnabled('createoff',canControl('create_power_off'));setEnabled('createoi',canControl('power_state'));setEnabledAll('[data-drive]',drive);setEnabled('speed',drive);setEnabled('turn',drive);base.style.pointerEvents=drive?'auto':'none';document.querySelectorAll('[data-action]').forEach(b=>{let v=actionVerb[b.dataset.action];b.disabled=!(v&&canControl(v))});setEnabled('undock',drive&&docked);setEnabled('dock',canMotion('dock'));setEnabled('silent',canSession('set_silent'));setEnabled('songdef',canControl('song_define'));setEnabled('songplay',canControl('song_play'));setEnabled('song',canControl('song_define')&&canControl('song_play'));setEnabled('songid',canControl('song_define')||canControl('song_play'));setEnabled('tones',canControl('song_define'));setEnabled('ping',hasVerb('ping'));setEnabled('refresh',true);refreshControlLock();if(caps&&caps.limits){if(caps.limits.max_linear_mm_s)$('speed').max=caps.limits.max_linear_mm_s;if(caps.limits.max_angular_mrad_s)$('turn').max=caps.limits.max_angular_mrad_s}}
function releaseDriveUi(){let wasDriving=active||timer||driveKind;clearInterval(timer);timer=0;active=false;driveKind='';nub.style.left='50%';nub.style.top='50%';document.querySelectorAll('[data-drive].active').forEach(b=>b.classList.remove('active'));return wasDriving}
function stop(){releaseDriveUi();sendCockpit({kind:'stop'})}
function clearLatch(kind,attempt=0){return sendCockpit({kind:'clear_safety_latch',latch:kind}).then(j=>{let reason=j&&(j.message||j.reason);if(j&&j.accepted===false&&reason==='busy'&&attempt<5){addLog('retry clear '+kind);return new Promise(r=>setTimeout(r,180)).then(()=>clearLatch(kind,attempt+1))}return requestStatus()})}
function joyMax(){return {lin:+$('speed').value,ang:+$('turn').value}}
function paceDrive(fn){let now=Date.now();if(now-lastDriveAt<120)return;lastDriveAt=now;fn()}
function refreshHeartbeat(){if(!hasVerb('heartbeat_stop'))return;let now=Date.now();if(now-lastHeartbeatAt>550){lastHeartbeatAt=now;sendCockpit({kind:'heartbeat_stop',timeout_ms:900},false)}}
function pulseCmdVel(lin,ang){if(!hasVerb('cmd_vel')){addLog('unsupported cmd_vel');stop();return}refreshHeartbeat();sendCockpit({kind:'cmd_vel',linear_mm_s:lin,angular_mrad_s:ang,ttl_ms:320},false)}
function undock(){releaseDriveUi();lastHeartbeatAt=Date.now();sendCockpit({kind:'cmd_vel',linear_mm_s:-1,angular_mrad_s:0,ttl_ms:10},false)}
function sendJoy(){paceDrive(()=>{let m=joyMax(),lin=Math.round(-last.y*m.lin),ang=Math.round(-last.x*m.ang);pulseCmdVel(lin,ang)})}
function sendDrive(){paceDrive(()=>{let m=joyMax(),lin=0,ang=0;if(driveKind==='fwd')lin=m.lin;if(driveKind==='back')lin=-m.lin;if(driveKind==='left')ang=m.ang;if(driveKind==='right')ang=-m.ang;if(driveKind==='spinl')ang=m.ang,lin=0;if(driveKind==='spinr')ang=-m.ang,lin=0;if(driveKind==='slow')lin=Math.round(m.lin*.45);pulseCmdVel(lin,ang)})}
function songSlot(){let n=parseInt($('songid').value,10);return Number.isFinite(n)?Math.max(0,Math.min(15,n)):0}
function defineSong(){return sendCockpit({kind:'song_define',id:songSlot(),tones:$('tones').value})}
function behavior(k){let v=actionVerb[k];if(v&&!hasVerb(v)){addLog('unsupported '+v);return}if(k==='heartbeat')sendCockpit({kind:'heartbeat_stop',timeout_ms:1200})}
function move(e){let r=base.getBoundingClientRect(),cx=r.left+r.width/2,cy=r.top+r.height/2,dx=e.clientX-cx,dy=e.clientY-cy,max=r.width*.34,d=Math.hypot(dx,dy);if(d>max){dx=dx/d*max;dy=dy/d*max}last={x:dx/max,y:dy/max};nub.style.left=(50+dx/r.width*100)+'%';nub.style.top=(50+dy/r.height*100)+'%';sendJoy()}
base.onpointerdown=e=>{active=true;base.setPointerCapture(e.pointerId);move(e);timer=setInterval(sendJoy,180)}
base.onpointermove=e=>{if(active)move(e)}
base.onpointerup=base.onpointercancel=stop
$('stop').onclick=stop;$('padstop').onclick=stop
$('careful').onclick=()=>sendCockpit({kind:'careful_mode',ttl_ms:5000}).then(()=>requestStatus())
$('sessionnew').onclick=establishBrowserSession
$('controllease').onclick=acquireBrowserControl
$('controlrefresh').onchange=syncControlRefresh
$('estop').onclick=()=>sendCockpit({kind:'estop'})
$('clear').onclick=()=>sendCockpit({kind:'clear_estop'})
$('clearcharge').onclick=()=>clearLatch('charging')
$('clearbump').onclick=()=>clearLatch('bump')
$('clearwheel').onclick=()=>clearLatch('wheel_drop')
$('clearcliff').onclick=()=>clearLatch('cliff')
$('cleartilt').onclick=()=>clearLatch('tilt')
$('clearimpact').onclick=()=>clearLatch('impact')
$('undock').onclick=undock
$('dock').onclick=()=>sendCockpit({kind:'dock'})
$('ping').onclick=()=>sendCockpit({kind:'ping'})
$('imuzero').onclick=()=>sendCockpit({kind:'zero_imu_orientation'}).then(requestStatus)
$('imuclear').onclick=()=>sendCockpit({kind:'clear_imu_orientation'}).then(requestStatus)
$('createrestart').onclick=()=>serviceCommand('restart_create',{kind:'restart_create'})
$('mbreset').onclick=()=>serviceCommand('reset_motherbrain',{kind:'reset_motherbrain'})
$('createon').onclick=()=>sendCockpit({kind:'create_power_on'})
$('createoff').onclick=()=>sendCockpit({kind:'create_power_off'})
$('createoi').onclick=()=>sendCockpit({kind:'power_state',request:'start_oi'})
$('silent').onchange=()=>{let silent=$('silent').checked;sendCockpit({kind:'set_silent',silent}).then(j=>{if(j&&j.accepted!==false)return requestStatus();$('silent').checked=!!(lastStatus&&lastStatus.audio_silent)})}
$('songdef').onclick=defineSong
$('songplay').onclick=()=>sendCockpit({kind:'song_play',id:songSlot()})
$('song').onclick=()=>defineSong().then(()=>sendCockpit({kind:'song_play',id:songSlot()}))
$('stream').onclick=()=>sendCockpit({kind:'stream_sensors',enabled:true,packet_id:0,period_ms:250})
$('bootsel').onclick=()=>serviceCommand('bootsel',{kind:'bootsel'})
$('refresh').onclick=connectSse
document.querySelectorAll('[data-action]').forEach(b=>b.onclick=()=>behavior(b.dataset.action))
document.querySelectorAll('[data-drive]').forEach(b=>{b.onpointerdown=e=>{driveKind=b.dataset.drive;b.classList.add('active');sendDrive();timer=setInterval(sendDrive,190);b.setPointerCapture(e.pointerId)};b.onpointerup=b.onpointercancel=stop})
$('speed').oninput=()=>$('speedv').textContent=$('speed').value
$('turn').oninput=()=>$('turnv').textContent=$('turn').value
function time(ms){let s=Math.floor((ms||0)/1000),m=Math.floor(s/60),h=Math.floor(m/60);return h+'h '+(m%60)+'m '+(s%60)+'s'}
function flagList(cs){let f=[],dock=homeBaseContact(cs);if(dock)f.push('Home Base contact');if(!dock&&cs.bump_left)f.push('bump L');if(!dock&&cs.bump_right)f.push('bump R');if(cs.wall)f.push('wall');if(cs.virtual_wall)f.push('virtual wall');if(cs.wheel_drop)f.push('wheel drop');if(cs.overcurrent)f.push('wheel overcurrent');if(!dock&&cs.cliff_left)f.push('cliff L');if(!dock&&cs.cliff_front_left)f.push('cliff FL');if(!dock&&cs.cliff_front_right)f.push('cliff FR');if(!dock&&cs.cliff_right)f.push('cliff R');return f}
function battPct(cs){return cs.capacity_mah?Math.min(100,Math.round((cs.charge_mah||0)*100/cs.capacity_mah)):null}
function num(v,d=0){return typeof v==='number'&&isFinite(v)?v.toFixed(d):'--'}
function pctBar(id,value,max,badAt,warnAt){let e=$(id),i=e&&e.querySelector('i');if(!e||!i)return;let p=Math.max(0,Math.min(100,(value||0)*100/max));i.style.width=p+'%';e.className='bar '+((value||0)>=badAt?'bad':(value||0)>=warnAt?'warn':'')}
function imuClass(imu){let h=imu.health||'unknown',age=imu.sample_age_ms||0;if(h==='fault'||(h==='ok'&&age>2000))return'badtext';if(h!=='ok'||age>500)return'muted';return'oktext'}
function showImu(imu){imu=imu||{};let present=imu.present||'unknown',health=imu.health||'unknown',age=imu.sample_age_ms||0,poll=imu.poll_period_ms||0,yaw=(imu.yaw_mrad||0)/1000,pitch=(imu.pitch_mrad||0)/1000,roll=(imu.roll_mrad||0)/1000,rate=(imu.yaw_rate_mrad_s||0)/1000,acc=(imu.accel_magnitude_mm_s2||0)/1000,tilt=(imu.tilt_magnitude_mrad||0)/1000,rough=(imu.roughness_mm_s2||0)/1000,impact=(imu.impact_score_mm_s2||0)/1000,av=imu.angular_velocity_mrad_s||{},la=imu.linear_acceleration_mm_s2||{};let cls=imuClass(imu);$('imuhealth').textContent=title(health)+' / '+title(present)+' / samples '+(imu.sample_count||0)+' / age '+age+' ms / '+poll+' ms poll';$('imuhealth').className=cls;$('imuyaw').textContent='yaw '+num(yaw,2)+' / pitch '+num(pitch,2)+' / roll '+num(roll,2)+' rad';$('imuyaw').className=cls;$('imuaccel').textContent=num(acc,2)+' m/s\u00B2 / xyz '+num((la.x||0)/1000,2)+','+num((la.y||0)/1000,2)+','+num((la.z||0)/1000,2);$('imuaccel').className=acc>16?'badtext':acc>12?'muted':cls;$('imutilt').textContent=num(tilt,2)+' rad / '+num(tilt*57.2958,1)+' deg';$('imutilt').className=tilt>.65?'badtext':tilt>.35?'muted':cls;$('imurates').textContent='yaw '+num(rate,2)+' rad/s / xyz '+num((av.x||0)/1000,2)+','+num((av.y||0)/1000,2)+','+num((av.z||0)/1000,2);$('imurates').className=cls;$('imurough').textContent=num(rough,2)+' m/s\u00B2';$('imurough').className=rough>8?'badtext':rough>3?'muted':cls;$('imuimpact').textContent=num(impact,2)+' m/s\u00B2';$('imuimpact').className=impact>18?'badtext':impact>8?'muted':cls;$('imumotion').textContent=title(imu.motion_consistency||'unknown')+' / '+title(imu.calibration||'uncalibrated');$('imumotion').className=(imu.motion_consistency==='inconsistent'||imu.calibration==='uncalibrated')?'muted':cls;pctBar('imuaccelbar',imu.accel_magnitude_mm_s2||0,22000,18000,13000);pctBar('imutiltbar',imu.tilt_magnitude_mrad||0,1000,650,350);pctBar('imuroughbar',imu.roughness_mm_s2||0,12000,8000,3000);pctBar('imuimpactbar',imu.impact_score_mm_s2||0,22000,18000,8000)}
function showStatus(s){lastStatus=s;let cs=s.create_sensors||{},od=s.odometry||{},imu=s.imu||{},music=s.create_songs||{},fatal=s.current_runtime_state==='error'||(s.last_error&&s.last_error!=='none'),dock=homeBaseContact(cs),contact=!dock&&(cs.bump_left||cs.bump_right||cs.wall||cs.virtual_wall),charging=chargeActive(cs),imuOk=imu.health==='ok',imuDanger=imu.health==='fault'||(imuOk&&((imu.tilt_magnitude_mrad||0)>=650||(imu.impact_score_mm_s2||0)>=18000)),safetyStop=s.estop_latched||s.safety_tripped||s.motion_interlock_latched||charging||cs.wheel_drop||(!dock&&(cs.cliff_left||cs.cliff_front_left||cs.cliff_front_right||cs.cliff_right))||imuDanger,pct=battPct(cs),flags=flagList(cs),latchKind=s.safety_latch_kind&&s.safety_latch_kind!=='none'?s.safety_latch_kind:'';if(s.estop_latched)flags.push('e-stop');if(s.safety_tripped)flags.push(latchKind?title(latchKind)+' latch':'safety latch');if(s.motion_interlock_latched)flags.push('charge latch');if(charging)flags.push('charging');if(imuOk&&(imu.tilt_magnitude_mrad||0)>=650)flags.push('tilt');if(imuOk&&(imu.impact_score_mm_s2||0)>=18000)flags.push('impact');if(imuOk&&imu.motion_consistency==='inconsistent')flags.push('motion mismatch');let safetyText=flags.join(', ')||'clear';pill(net,wsOpen?'control ws':sseOpen?'telemetry sse':(s.wifi_state||'online'),'ok');pill($('mode'),title(s.oi_mode),(s.oi_mode==='safe'||s.oi_mode==='full')?'ok':'');pill($('safety'),safetyStop?'motion blocked':dock?'Home Base':contact?'contact':'clear',safetyStop?'bad':(dock||contact)?'warn':'ok');$('headline').textContent=title(s.current_runtime_state)+' / '+title(s.create_power_state)+' / '+title(s.uart_rx_health)+' / IMU '+title(imu.health||'unknown');$('runtime').textContent=title(s.current_runtime_state)+' / body '+title(s.body_state);$('uptime').textContent=time(s.uptime_ms);$('create').textContent=title(s.create_power_state)+' / '+title(s.oi_mode)+' / probe '+s.wake_probe_response_bytes+'/'+s.wake_probe_expected_bytes;$('safetyread').textContent=safetyText;$('safetyread').className=safetyStop?'badtext':(dock||contact)?'muted':'oktext';$('uart').textContent=title(s.uart_rx_health)+' / '+title(s.last_uart_read_error)+' / '+s.uart_rx_packets+' packets';$('cmd').textContent=title(s.current_command)+' / pending '+title(s.pending_command)+' #'+s.pending_command_id;$('forebrain').textContent=(s.forebrain_uart?s.forebrain_uart.rx_lines:0)+' lines / '+title(s.forebrain_uart&&s.forebrain_uart.last_error);$('web').textContent=s.http_requests+' requests / '+s.dhcp_grants+' dhcp';$('sensors').textContent='pkt '+(cs.last_packet_id||0)+' / IR '+(cs.ir_byte||0)+' / buttons '+(cs.buttons||0)+' / cliff sig '+(cs.cliff_left_signal||0)+','+(cs.cliff_front_left_signal||0)+','+(cs.cliff_front_right_signal||0)+','+(cs.cliff_right_signal||0);$('battery').textContent=(pct===null?'--':pct+'%')+' / '+(cs.voltage_mv||0)+' mV / '+(cs.current_ma||0)+' mA / '+(cs.charge_mah||0)+'/'+(cs.capacity_mah||0)+' mAh / charge state '+(cs.charging_state||0)+' / charge pin '+title(cs.charging_indicator);$('battery').className=(charging||pct!==null&&pct<=20)?'badtext':'muted';$('odom').textContent='delta '+(cs.distance_mm||0)+' mm / '+(cs.angle_mrad||0)+' mrad / total '+(od.distance_mm||0)+' mm / '+(od.heading_mrad||0)+' mrad / resets '+(od.reset_count||0);showImu(imu);$('silent').checked=!!s.audio_silent;$('music').textContent=(s.audio_silent?'silent / ':'audible / ')+'defined '+(music.last_defined_id||0)+' ('+(music.last_defined_len||0)+') / played '+(music.last_played_id||0);$('firmware').textContent=s.firmware_name+' '+s.firmware_version;$('err').textContent=fatal?title(s.last_error)+' / '+(s.last_error_hint||''): 'none';$('err').className=fatal?'badtext':'muted';applyCaps()}
function handleEvents(batch){let stopNeeded=false,refreshNeeded=false;eventCursor=Math.max(0,(batch.next_seq||1)-1);if(batch.dropped_before_seq){$('events').textContent='recovered after '+batch.dropped_before_seq;pill($('safety'),'event history recovered','warn');addLog('recovered event history after '+batch.dropped_before_seq);stopNeeded=true}else{$('events').textContent='cursor '+(batch.next_seq||0)+' / '+((batch.events||[]).length)+' new'}(batch.events||[]).forEach(e=>{let k=e.kind;if(['safety_tripped','heartbeat_expired','estop_latched','wheel_drop_latched'].indexOf(k)>=0){pill($('safety'),title(k),'bad');addLog('safety '+k+' '+(e.a||0));stopNeeded=true;refreshNeeded=true}else if(k==='safety_cleared'){pill($('safety'),'clear','ok');$('safetyread').textContent='clear';$('safetyread').className='oktext';addLog(k+' '+(e.a||0));refreshNeeded=true}else if(['imu_frame_received','imu_fault','tilt_changed','impact_detected','imu_calibration_changed'].indexOf(k)>=0){addLog(k+' '+(e.a||0));refreshNeeded=true}else if(['bump_changed','wall_changed','virtual_wall_changed','buttons_changed','ir_changed','charging_state_changed','battery_low','cliff_changed','wheel_drop_cleared'].indexOf(k)>=0){addLog(k+' '+(e.a||0));refreshNeeded=true}else if(['command_rejected','command_interrupted'].indexOf(k)>=0){pill($('safety'),title(k),'warn');addLog(k+' #'+(e.a||0));refreshNeeded=true}else if(k==='motion_stopped'){addLog('motion stopped')}else if(k==='error'){let message=errorName(e.a||0);pill($('safety'),message,'bad');addLog(message);stopNeeded=true;refreshNeeded=true}});if(refreshNeeded)requestStatus();if(stopNeeded&&releaseDriveUi())sendCockpit({kind:'stop'})}
const renderStatus=showStatus;
showStatus=function(s){renderStatus(s);$('firmware').textContent=(s.build_id||((s.firmware_name||'')+' '+(s.firmware_version||'')))+' / '+(s.git_commit_short||'unknown')+(s.git_dirty?' dirty':'');let careful=!!s.careful_mode_active,remaining=s.careful_mode_remaining_ms||0;$('careful').classList.toggle('active',careful);if(careful){pill($('safety'),'CAREFUL '+(remaining/1000).toFixed(1)+' s','warn');$('safetyread').textContent='CAREFUL: sensor gates advisory / '+$('safetyread').textContent;$('safetyread').className='muted'}}
applyCaps();establishBrowserSession().catch(connectSse);
</script>
</body>
</html>
"#
}
