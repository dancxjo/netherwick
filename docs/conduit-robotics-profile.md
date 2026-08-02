# Conduit audited describe-only robotics profile

`pete_brainstem::conduit_robotics` publishes the current Pete robotics profile
through two real, effect-free implementations:

- `netherwick/implementation/pete-linux-robotics-describe`
- `netherwick/implementation/pete-pico-robotics-describe`

Both implement the same `conduit.robotics/profile` contract and carry the same
canonical profile identity. Their implementation, artifact, provider bundle,
host profile, target, source build, and compiled-capability facts are distinct.
These are ordinary Conduit implementation facts; there is no Netherwick-only
registry, resolver, runtime, or event model.

The public `linux_report()`, `pico_w_report()`, and `describe_only()` calls only
assemble compile-time and source-owned data. They do not open Create UART, USB,
sensors, CYW43, sockets, cockpit transports, or higher-brain services. They do
not observe a boot, initialize a provider, discover or admit a carrier path,
enroll an entity, acquire possession or authority, promote Katra, load or
promote an Organism Runtime checkpoint, activate a plan, or actuate.

## Semantic separation

The report carries independent exact identities for observation, command,
acknowledgement, safe outcome, possession, terminal, and fault values. An
observation cannot be supplied as a command. Acknowledgement does not claim a
safe physical outcome. Terminal and fault remain separate.

Each physical quantity has its own units, frame, brainstem monotonic clock,
uncertainty bound, and maximum age. The finite motion envelope separately pins
command TTL, linear and angular velocity, queue capacity, possession, motion
authority, stop, emergency stop, not-charging state, charging interlock, clear
safety inhibit, and exact motion capability requirements.

Stable `motherbrain-to-brainstem` and `brainstem-to-body` logical
relationships list possible carrier kinds. They are not current paths. A
describe-only report always has an empty `current_path_observations` array, and
every carrier candidate has `admitted: false`.

Entity, boot, role, possession, and authority are separate fields. The report
publishes the stable entity and Katra role descriptors while leaving boot,
possession, and authority absent. It names the Organism Runtime continuation
descriptor with `loaded: false` and `promoted: false`.

Compiled and initialized provider states are also separate. USB, CYW43,
network, and Create-control candidates say either `compiled` or `unsupported`;
their `initialized_observation` remains absent. Live networking continues to
be owned by `conduit_network::runnable(now_ms)`, which obtains its own fresh
firmware observation after actual initialization.

## Redaction and evidence

Reports contain no raw device handles, credentials, bearer tokens, private
endpoints, or sensitive topology. `DescribeEffectAudit` contains separate
counts for device opens, network joins, relays, possession, role promotion,
plan activation, and actuation; all are zero.

`SOURCE_EVIDENCE` maps body/build, observation, command, outcome, sensors,
possession, networking, Linux control, higher-brain role, and cockpit session
facts to their current source and focused tests. The exact cross-repository
profile and host hashes are asserted in both Netherwick's
`conduit_robotics::tests` and Conduit's `conduit-robotics` conformance tests.

Physical equivalence, live provider observations, execution, carrier
switching, role promotion, checkpoint loading, and actuation remain outside
this describe-only profile.
