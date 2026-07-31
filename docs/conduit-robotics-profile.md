# Conduit describe-only robotics profile

`pete_brainstem::conduit_robotics::describe_only()` is the auditable current
inventory for the installed Create Open Interface brainstem build. It returns
compiled descriptors only: it does not open Create UART, sensors, CYW43,
network sockets, cockpit transports, a higher-brain provider, or a possession
lease, and it cannot arm or actuate the robot.

The artifact names nine distinct boundaries: compiled brainstem/body
description; Create observations; bounded motion commands; acknowledgements
and safe outcomes; sensor observations; possession leases; network
observations; higher-brain providers; and cockpit sessions. Each descriptor
declares its units, frame, monotonic clock, uncertainty, freshness, authority,
effect, host requirement, and exact source surface. They remain descriptions,
not claims that any provider is live or authorized.

In particular, a possession lease is neither a motor grant nor a safety
override; network attachment is neither enrollment nor possession; command
acceptance is not a safe outcome; and a status observation is never a command.
Actual host reports must be created from fresh initialized host state and
selected with a separate exact plan. Hardware equivalence and actuation remain
outside this profile.

The inventory is intentionally small and allocator-free so both Pico W and
Linux candidates can describe the same domain vocabulary while preserving their
different build, device, driver, resource, and current-observation facts.

## Source and test evidence

| Boundary | Current source | Deterministic check |
| --- | --- | --- |
| body/brainstem description | `capabilities.rs`, `body.rs` | `conduit_robotics::describe_only_inventory_is_effect_free_and_complete` |
| Create observation and command | `runtime/lifecycle.rs`, `runtime/motion.rs` | brainstem runtime/safety vectors |
| safe outcome | `runtime/execution.rs`, `events.rs` | brainstem runtime/safety vectors |
| sensor observation | `runtime/sensors.rs`, `drivers/imu.rs` | brainstem status and IMU vectors |
| possession | `pete-cockpit/src/cockpit/possession` | cockpit possession/session vectors |
| isolated network | `conduit_network.rs` | `conduit_network` unit vectors |
| higher brain | `pete-higher-brain/src/capability.rs` | higher-brain capability vectors |
| cockpit | `pete-cockpit` protocol/session modules | cockpit contract/session vectors |

`conduit_robotics::observation_command_and_safe_outcome_are_not_one_boundary`
and `conduit_robotics::no_boundary_turns_network_or_possession_into_motion_authority`
enforce the two critical separation invariants directly.
