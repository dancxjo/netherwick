# Motherbrain routed network

These files are an optional fixed-interface profile for Pete's three routed
failure domains. They preserve the original direct-Ethernet installation but
are not required by the higher-brain protocol. Portable service selection is
configured in `configs/higher-brain/*.toml` and excludes the brainstem
interface by default.

- `wlan0`: external infrastructure; never an identity source.
- `wlan1`: Pico W management subnet, DHCP client.
- `eth0`: independent interbrain backbone at `10.42.0.1/24`.

Install the sysctl and nftables files through the image/configuration manager.
Install `pete-interbrain-dnsmasq.conf` under the dnsmasq configuration directory
and `pete-motherbrain.service` under `/etc/avahi/services`.
Do not create a Linux bridge between these interfaces. Forwarding is
stateful/default-deny and exposes only discovery and cockpit services from the
interbrain network toward the brainstem subnet.

The brainstem AP is also the last-resort host-transit segment. Motherbrain's
`wlan1` DHCP address is registered as `motherbrain.pete.internal` through its
identified brainstem session. Port 8787 on that interface exposes only bounded
peer health, role coordination, and handback; the higher-brain data plane still
excludes `wlan1`, and SSH, model, experience, perception, package, and update
traffic remain unavailable there. Joining the AP does not grant possession.
