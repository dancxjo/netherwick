# Solresol auditory cues

Pete's auditory annunciation is owned by the Brainstem runtime. The cockpit can
configure it and display its status, but physical edge detection, priority,
deduplication, and Create OI playback remain beside the safety and body state
that they describe.

These motifs use only the seven diatonic Solresol pitch classes: do, re, mi,
fa, sol, la, and si. They are stable operational codes, not claims that every
sequence is a grammatical Solresol word. Octave placement is an implementation
detail. `fa-sol-si` remains Pete's established prepare / make-ready cue.

## Vocabulary

| Cue | Motif | Meaning |
| --- | --- | --- |
| Armed | fa-sol-si | Create OI reached verified Full readiness |
| E-stop | do-do-do-do | E-stop latched |
| Cliff | si-do-si-do | Cliff hazard |
| Wheel drop | do-fa-do-fa | Wheel-drop hazard |
| Tilt | la-fa-re | Sustained unsafe tilt |
| Impact | do-sol-do | Impact threshold crossed |
| Heartbeat lost | sol-mi-do | Motion heartbeat expired |
| Authority lost | sol-re-do | Control lease expired |
| Authority replaced | sol-fa-re | A different controller replaced authority |
| Bump/contact | do-re-do | Contact withdrawal or stationary bump stop |
| Create error | mi-re-do-do | Create link became unresponsive |
| Runtime error | mi-re-do | Brainstem runtime entered error |
| Service failure | fa-re-do | Verified service/reflex terminal failure |
| Low battery | la-mi-do | Battery crossed into the low band |
| Safety clear | do-mi-sol | A latched safety condition was cleared |
| Recovery | re-fa-la | Stable IMU or Create recovery |
| Authority acquired | do-sol-mi | A controller acquired authority |
| Dock contact | sol-si-re | Charging or Home Base contact became true |
| IMU fault | si-fa-mi | IMU entered fault or absent health |
| Service complete | re-sol-si | Verified reset or Create restart completion |
| Dock seen | sol-re-si | Home Base IR was newly acquired while docking |
| Motion inconsistency | mi-si-mi | IMU/motion evidence became inconsistent |

The six configurable feedback motifs retain their public names and now use the
same pitch vocabulary: OK `do-mi-sol`, Error `mi-re-do`, Armed `fa-sol-si`, Lost
target/control `sol-mi-do`, Dock seen `sol-si-re`, and Danger `do-do-do`.

## Priority and bounded scheduling

The scheduler retains at most one urgent cue and one newest informational cue.
It never builds a history. A newer informational cue replaces the older one;
an urgent cue discards queued information; and a stronger urgent cue replaces a
weaker pending urgent cue. The effective order is:

1. E-stop.
2. Cliff, wheel drop, tilt, and impact.
3. Heartbeat, authority, and Create-link loss.
4. Bump/contact withdrawal.
5. Runtime, service, IMU, and motion-consistency errors.
6. Low battery.
7. Safety clear and verified recovery.
8. Authority acquired, docking, ready, and service-complete information.

Create OI song playback is asynchronous. The Brainstem computes the bounded
motif duration and does not issue another play request until it expires. Manual
feedback and defined-song playback share that busy window. A playing motif is
allowed to finish; it is not interrupted to simulate preemption. Audio driver
failures are counted and discarded without changing stop, safety, watchdog, or
runtime control behavior.

## Edge detection and anti-chatter

Only transitions sound:

- bump, cliff, wheel-drop, tilt, and impact use the existing safety latch and
  contact-withdrawal edges, so held sensor frames do not repeat;
- heartbeat expiry clears its deadline, so it sounds once per expiry;
- authority acquisition, replacement, and expiry use the authority barrier;
- low battery sounds on entry at 20 percent and re-arms above a 25 percent
  hysteresis boundary;
- charging/Home Base contact sounds on the false-to-true transition;
- IMU fault or absence sounds on entry, while recovery requires 500 ms of
  continuously verified OK health;
- motion inconsistency is edge-triggered with a five-second cooldown;
- Create loss sounds on responsive-to-unresponsive, and recovery/ready sounds
  only after Full mode is observed again;
- dock IR sounds only when a nonzero IR character is newly acquired during an
  active seek-dock operation;
- motherbrain reset sounds at terminal pulse completion or refusal, and Create
  restart completion waits for Full readiness.

Routine commands, command renewal, normal TTL completion, repeated sensor
frames, odometry, wall and virtual-wall observations, DHCP, DNS, HTTP, ICMP,
and ordinary network plumbing remain silent. BOOTSEL success cannot sound
because successful entry transfers execution away from the running firmware;
only terminal service states observable while the Brainstem remains running can
be annunciated.

## Silent mode and observability

Use either transport through the cockpit CLI:

```text
pete-cockpit audio silent
pete-cockpit audio audible
```

The command is `SET_SILENT true|false` on the compact USB/UART contract and
`set_silent` with a boolean `silent` field over HTTP. The Brainstem reads status
back through the CLI before reporting success. The setting is volatile and
returns to audible on Brainstem reboot.

While silent, automatic feedback and direct song playback are suppressed,
pending cues are discarded, and no cue is replayed after returning to audible.
Song and chirp definitions may still be updated. Safety behavior, events,
status, LEDs, and OLED alerts are unchanged.

Status exposes top-level `audio_silent` plus an `audio` object containing the
last requested cue, last played cue, last playback timestamp, silent-suppressed
count, and dropped/replaced count. State changes emit the edge-based
`audio_state_changed` event.
