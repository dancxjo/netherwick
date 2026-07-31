# Conduit network provider boundary

The Pico W brainstem publishes four bounded network capabilities to Conduit:
`net/wifi/access-point`, `net/dhcp/server`, `net/reachability`, and
`net/dns-sd`. The firmware remains the implementation owner; Conduit owns the
semantic contracts and exact plans that select them.

`pete_brainstem::conduit_network::describe_only()` reports compiled inventory
and exact firmware build identity without initializing hardware or claiming a
live observation. `runnable(now_ms)` reports an initialized observation only
after the real CYW43 access point and all four service tasks have started. It
reads availability, device identity, boot identity, and interface generation
from firmware-owned state. If the Wi-Fi lane exits or its status stops being
ready, no runnable observation is returned.

The adapter exposes no possession, motor, safety, grant, destination, or other
robot authority. It also exposes no public setter through which a caller can
manufacture provider availability or freshness. The access point is explicitly
isolated: no routing, bridging, or NAT.

The corresponding Conduit ownership/conformance fixture is
`conformance/c4/netherwick-network.json`, with runnable examples under
`examples/{wifi-ap-isolated,dhcp-server,icmp-reachability,dns-sd-local}.panel`.
