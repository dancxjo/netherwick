# Motherbrain routed network

These files describe Pete's three routed failure domains. Interface names are
deployment defaults and should be overridden with udev/systemd `.link` files
when adapter MACs are known.

- `wlan0`: external infrastructure; never an identity source.
- `wlan1`: Pico W management subnet, DHCP client.
- `eth0`: independent interbrain backbone at `10.42.0.1/24`.

Install the sysctl and nftables files through the image/configuration manager.
Install `pete-interbrain-dnsmasq.conf` under the dnsmasq configuration directory
and `pete-motherbrain.service` under `/etc/avahi/services`.
Do not create a Linux bridge between these interfaces. Forwarding is
stateful/default-deny and exposes only discovery and cockpit services from the
interbrain network toward the brainstem subnet.
