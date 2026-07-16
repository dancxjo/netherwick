# Brainstem boundary hardware-in-loop checklist

This checklist records the physical evidence required before issue #78 can be
closed. Run with wheels safely clear unless a step explicitly requires a
controlled contact surface. Preserve the event transcript, status snapshots,
firmware build identity, command timestamps, and final wheel output for every
case.

- Boot with left, right, then both bumpers held: observe contact evidence and
  no reverse command.
- Press each bumper while stationary: observe no autonomous motion.
- Drive forward with 250 ms `cmd_vel` renewals into left, right, then bilateral
  contact: verify command interruption within one 10 ms runtime cycle, straight
  bounded withdrawal, typed started/completed events, then zero wheels.
- Remove the host link during withdrawal: verify withdrawal finishes within its
  time/distance bound and remains stopped.
- Introduce e-stop, wheel drop, cliff, charging/dock, and tilt/impact separately
  during withdrawal: verify immediate zero output and a typed safety-preempted
  reflex completion naming the dominating invariant.
- Attempt every retired HTTP, WebSocket, UDP, and UART verb: verify typed
  `unsupported` rejection and no motor output.
- Run one motherbrain bearing or approach skill: verify only short TTL
  primitives on the wire, successful renewal while inputs are fresh, motor
  zeroing after renewal stops, and skill `safety-preempted` after a contact
  reflex.

Do not record this checklist as passed from simulator or unit-test evidence.
