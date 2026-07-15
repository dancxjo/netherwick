# Brainstem body model

The brainstem is a deterministic controller for an embodied machine.

It is not inherently a robot controller, a motion controller, or an iRobot
Create adapter. The current implementation controls an iRobot Create because
that is Pete's first body. The architectural boundary is broader:

> A brainstem owns body-local hardware, immediate sensing, actuator authority,
> safety enforcement, bounded command execution, and an ordered account of what
> physically happened.

Planning, task selection, mapping, language models, long-horizon autonomy, and
learning remain outside this boundary. A higher brain may request an action. It
may not bypass the brainstem's local authority over physical outputs.

## Why this boundary exists

Physical machines operate on timescales and failure modes that do not belong in
a general-purpose planning process.

A wheel must stop when its deadline expires. A pinball coil must not remain at
full current because a game-rules process paused. An oven must disable its
heaters when a temperature probe becomes implausible. A kettle must not
energize an empty vessel. These decisions must survive slow inference, process
restarts, network loss, malformed commands, and unavailable higher-level
software.

The brainstem therefore provides four guarantees:

1. **Local authority.** Only the brainstem's safety/runtime lane can directly
   affect the body.
2. **Bounded execution.** Physical actions have explicit limits, deadlines, or
   completion conditions.
3. **Fail-closed behavior.** Lost authority, stale commands, invalid state, and
   detected faults drive the body toward a body-defined safe state.
4. **Observable consequences.** Accepted requests, actual starts, completion,
   interruption, timeout, sensor changes, and safety transitions are recorded
   as ordered events.

## Layers

```text
motherbrain / forebrain / human operator
    plans, reasons, learns, chooses goals
                    |
                    | body-neutral requests and events
                    v
cockpit contract and transports
    validates advertised capabilities and carries requests
                    |
                    v
brainstem safety/runtime lane
    owns authority, deadlines, queues, reflexes, and safe transitions
                    |
                    v
body driver
    maps typed actions and sensors onto one physical machine
                    |
                    v
board backend and electrical hardware
    GPIO, UART, I2C, SPI, PWM, ADC, relays, drivers, watchdogs
```

### Higher brain

The higher brain decides what should be attempted and why. It may sequence
behaviors, interpret perception, maintain a world model, learn, or ask for a
fresh action pulse in a feedback loop. It does not own actuator handles.

### Cockpit contract

The cockpit contract is the body-neutral control surface visible to pilots.
Pilots may be humans, simulations, deterministic controllers, learned
controllers, or language-model-mediated processes. They discover the active
body's capabilities and reject unsupported operations before issuing them.

Transport is not authority. HTTP, WebSocket, UDP, UART, or another future
transport carries a request into a bounded queue. No transport lane receives
direct access to body GPIO, buses, relays, or motor commands.

### Brainstem runtime

The runtime validates authority and command shape, accepts or rejects the
request, advances bounded work, samples local sensors, applies safety policy,
and emits lifecycle events. It is tick-driven and uses fixed-capacity state so
that safety behavior does not depend on allocation or an unbounded queue.

### Body driver

A body driver translates the generic runtime contract into the protocol and
physics of one machine. For the Create body this includes Open Interface
opcodes, OI modes, sensor packets, BRC, LEDs, songs, docking, and differential
drive. Other bodies would own different operations without changing the reason
for the brainstem boundary.

### Board backend

The board backend owns chip and board details such as pin assignments, UART
instances, I2C peripherals, PWM channels, watchdog configuration, and voltage or
polarity assumptions. The body and board are separate choices. A body should
not silently acquire new semantics because it moved from an RP2040 to another
microcontroller.

## Body declaration

`body.toml` is the selected body's declarative contract. It identifies the body
kind and describes the verbs, events, sensors, outputs, safety features, timing,
and limits that the firmware advertises.

`board.toml` describes the controller board and physical mapping.

The firmware currently generates static, allocation-free body constants at
build time. This is intentional: the embedded runtime does not discover its own
safety contract by parsing mutable configuration after boot.

A complete body implementation consists of:

- a body kind and descriptor;
- a capability declaration;
- typed commands or actions;
- a body driver;
- sensor decoding and freshness rules;
- actuator implementations and safe-output transitions;
- local reflex and safety policy;
- lifecycle and body-specific outward events;
- simulator or fake-hardware coverage;
- bounded hardware bring-up tests.

## Action lifecycle

The generic lifecycle is:

```text
request arrives
    -> validate authority, capability, state, arguments, and limits
    -> accept or reject
    -> queue or preempt according to action class
    -> start when the runtime actually begins physical execution
    -> sample sensors and enforce local invariants every tick
    -> complete, interrupt, or time out
    -> record the transition as an ordered outward event
```

These terms are deliberately precise:

- **accepted** means the request passed validation and entered the runtime's
  responsibility;
- **started** means the runtime began executing it;
- **completed** means its declared work or ordinary duration ended normally;
- **interrupted** means stop, emergency stop, replacement, reflex, or policy
  ended it before normal completion;
- **timed out** means a required response or operation failed its deadline;
- **rejected** means execution never began.

Ordinary expiry of a bounded action is completion, not failure. A 250 ms drive
pulse that stops at 250 ms has done exactly what was requested.

## Safety is body-defined, not merely "all outputs off"

Every body needs a safe-output transition. That transition is not necessarily
zero output.

For Pete, safety normally means stopping wheel motion while retaining enough
supervision to report state and recover. For an oven, cancellation disables the
heaters but may require a cooling fan to continue. For a pinball machine, a
slam tilt disables high-energy coils while status lamps and switch scanning may
remain active. For a kettle, the heater turns off immediately while sensing and
fault indication continue.

The body driver and safety policy therefore define:

- which outputs are hazardous;
- which outputs must preempt immediately;
- which outputs may continue during a safe transition;
- which faults latch until explicit clearing or physical intervention;
- which actions require a heartbeat, lease, or fresh sample stream;
- which sensors gate actuation;
- what must happen on startup, restart, transport loss, and watchdog reset.

## Reflexes and higher-level feedback

A reflex is a body-local response whose timing or reliability cannot depend on
a higher process.

Examples include stopping for a cliff, cutting power to an overheating coil,
disabling an oven heater after a probe fault, or ending kettle heat when boil is
detected. The brainstem may also expose one-shot controller primitives that use
a fresh error or range sample supplied by the host. Those primitives remain
bounded. Continuous closed-loop behavior requires the host to keep sending
fresh observations or targets.

This preserves a useful division:

- the brainstem owns immediate consequences and invariants;
- the higher brain owns interpretation, goals, adaptation, and long-horizon
  control.

## Ordered events are the physical transcript

Commands describe intention. Events describe what the runtime claims occurred.

The event log is sequence-numbered and finite. A consumer that requests history
older than the retained ring is told that records were dropped. It must not
quietly continue as though its physical transcript were complete.

A body implementation should expose generic lifecycle and safety events wherever
possible, then add body-specific vocabulary for facts that matter to pilots.
Examples include sensor transitions, actuator inhibition, current or thermal
faults, target attainment, vessel presence, or ball-trough count.

## Worked body: iRobot Create

The current body is an iRobot Create using Open Interface over UART.

The brainstem owns:

- Create power and interface supervision;
- OI mode acquisition and maintenance;
- differential-drive requests and deadlines;
- immediate stop and emergency stop;
- bump, cliff, wheel-drop, tilt, impact, and heartbeat safety;
- Create sensor decoding and lightweight odometry accumulation;
- local feedback through lights, songs, and chirps;
- ordered command, sensor, motion, safety, and authority events.

The motherbrain owns perception, mapping, planning, learned behavior, and the
choice to request a new bounded body action. It does not write Create opcodes.

## Worked body: pinball machine

A pinball brainstem is a real-time playfield controller.

It would own:

- switch-matrix scanning;
- flipper, bumper, slingshot, kicker, ejector, and reset coils;
- lamp and display outputs assigned to the body layer;
- ball-trough and shooter-lane sensing;
- tilt and slam switches;
- coil current, pulse duration, duty cycle, and temperature limits.

Representative verbs might include:

```text
arm_playfield
stop_all_coils
fire_coil
set_flipper
reset_drop_targets
eject_ball
serve_ball
set_lamp
run_lamp_pattern
estop
```

The higher game process decides that a target sequence begins multiball and
awards points. The brainstem reacts to time-critical switch closures, performs
bounded coil pulses, applies hold current where permitted, prevents mutually
unsafe outputs, and refuses to cook a stuck coil.

Representative events might include:

```text
switch_closed
switch_opened
coil_pulsed
coil_inhibited
ball_served
trough_count_changed
tilt_warning
tilt_latched
coil_overcurrent
coil_overtemperature
```

This body demonstrates why game rules and physical reflexes must be separate.
A paused or restarted scoring process must not change the electrical limits of
a flipper coil.

## Worked body: countertop toaster oven

An oven brainstem is a thermal process controller with final veto authority.

It would own:

- upper and lower heating elements;
- convection and cooling fans;
- temperature, current, door, and optional food-probe sensing;
- heater duty-cycle limits;
- overtemperature and sensor-plausibility policy;
- stage timing and safe cooldown behavior.

Representative verbs might include:

```text
arm_heat
disarm_heat
set_temperature
set_power
set_element_balance
set_fan
run_profile
pause_profile
resume_profile
cancel_profile
cool_down
estop
```

A higher process may select a recipe, infer food type, or request a sequence of
preheat, bake, broil, and cooldown stages. The brainstem decides whether heat is
currently permissible, whether the door or sensors invalidate the request,
whether temperature response is plausible, and whether cooling must continue
after cancellation.

Representative events might include:

```text
door_opened
door_closed
preheat_started
target_reached
stage_started
stage_completed
heater_enabled
heater_inhibited
probe_fault
temperature_runaway
heating_ineffective
cooldown_started
safe_temperature_reached
```

This body demonstrates that a safe transition can require continuing one
actuator after disabling another.

## Worked body: electric kettle or teapot

An electric kettle is the smallest clear expression of the model.

It would own:

- the heater relay or triac;
- water temperature and boil sensing;
- vessel, lid, level, weight, or dry-boil inputs where available;
- automatic shutoff;
- maximum heating duration;
- status light and chime.

Representative verbs might include:

```text
heat_to
boil
hold_temperature
cancel
stop
estop
set_feedback
```

The higher process may know that one tea wants 80 degrees Celsius and another
wants water near boiling. The brainstem knows whether the vessel is present,
whether heat is safe, whether boil has occurred, and whether a sensor or empty
vessel requires an immediate latched stop.

Representative events might include:

```text
vessel_attached
vessel_removed
water_detected
heating_started
target_reached
boil_detected
heater_stopped
dry_boil_detected
sensor_fault
keep_warm_started
keep_warm_expired
```

## Present implementation boundary

The architecture is partly generic and partly still shaped by its first body.

Already generic:

- body and board configuration are separate;
- capabilities are body-owned and rendered through a common surface;
- commands and events use fixed-capacity queues;
- the runtime is deterministic and tick-driven;
- transport lanes are separated from the safety/runtime lane;
- requests have explicit lifecycle events;
- pilots validate against the advertised capability contract;
- unsupported verbs fail closed;
- event consumers can detect missed history;
- the RP2040 and Pico W backends are distinct from Create-specific drivers.

Still Create-shaped:

- the central action vocabulary often says `motion` rather than `actuation` or
  `body action`;
- several public verbs are inherently differential-drive operations;
- limits emphasize linear speed, angular speed, and motion TTL;
- some status and internal event types retain Create-specific names;
- safe transition paths commonly assume that sending Stop is the principal
  physical consequence;
- body descriptors are generated constants rather than a stronger typed driver
  contract.

This is not a defect in the current Create implementation. It identifies the
seams that a second body must test.

## Generalization path

The next architectural steps should be driven by a real second body rather than
by speculative abstraction alone.

1. Introduce a generic body-action concept while preserving the existing Create
   motion verbs as Create capabilities.
2. Generalize runtime queue and lifecycle names where they currently assume
   movement, for example `clear_motion_queue`, `motion_requested`, and
   `motion_stopped`.
3. Define a body-owned safe-output transition instead of assuming that every
   failure reduces to a motor Stop command.
4. Promote generated body descriptors into a typed contract implemented by body
   drivers.
5. Generalize limits so bodies can advertise named quantities such as speed,
   duty cycle, current, temperature, pulse duration, hold duration, or action
   timeout.
6. Separate generic status from optional body diagnostic payloads.
7. Implement one deliberately non-mobile body as an architectural proof. A
   kettle simulator or small low-voltage thermal fixture would exercise
   actuation, sensing, target attainment, cooldown, and fault latching without
   introducing the electrical hazards of a mains appliance.
8. Keep the Create path stable and compatibility-tested while generic terms are
   introduced.

Possible future names include `BodyAction`, `ActuatorRequest`, or
`TimedOperation`. Naming should follow the type system and actual second-body
implementation, not precede them.

## Porting checklist

Before declaring a new body supported, answer all of the following:

### Contract

- What is the body kind?
- Which verbs are public?
- Which sensors, outputs, safety features, events, and limits are advertised?
- Which operations are one-shot, duration-bounded, target-bounded, or
  continuously refreshed?

### Authority

- Which physical handles are owned exclusively by the safety/runtime lane?
- Which actions require possession, a heartbeat, or another lease?
- Which commands preempt queues immediately?

### Safety

- What is the safe-output transition?
- Which faults latch?
- Which sensors gate each hazardous output?
- What happens on malformed input, stale input, transport loss, runtime error,
  reboot, and watchdog reset?
- Does safety require any output to continue temporarily?

### Physics

- What electrical, thermal, mechanical, and timing limits must be enforced
  locally?
- Which combinations of outputs are forbidden?
- How are missing, contradictory, stale, or implausible sensors handled?

### Observability

- Which lifecycle events prove that an action actually began and ended?
- Which body-specific events are needed to reconstruct important physical
  state?
- How does a pilot detect missed history or stale telemetry?

### Validation

- Is there a simulator or fake hardware implementation?
- Are deadlines, preemption, latches, and safe transitions tested?
- Is bring-up staged from non-actuating inspection to bounded physical tests?
- Is there a readily accessible operator stop path?

## Design rule

A useful test for every new feature is:

> Could a disconnected, restarted, confused, or malicious higher process cause
> this body to violate its local physical invariants?

If the answer is yes, the invariant or actuator authority belongs lower in the
brainstem boundary. If the answer is no and the behavior concerns goals,
meaning, learning, or long-horizon choice, it likely belongs above it.

The first body is Pete. The architecture is for embodied machinery.