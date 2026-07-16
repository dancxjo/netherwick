use crate::capability::AcceleratorCapabilities;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::net::{IpAddr, Ipv4Addr, SocketAddr, UdpSocket};
use std::path::Path;
use std::process::Command;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LinkKind {
    Ethernet,
    Wifi,
    Other,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InterfaceAddress {
    pub name: String,
    pub address: IpAddr,
    pub kind: LinkKind,
    pub up: bool,
    pub route_metric: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct DataPlaneConfig {
    /// If nonempty, only these interfaces may carry higher-brain traffic.
    pub allowed_interfaces: BTreeSet<String>,
    /// Bodily control-plane interfaces. They are denied even if otherwise usable.
    pub brainstem_interfaces: BTreeSet<String>,
    /// Explicit override for a lab where a shared interface is intentional.
    pub allow_brainstem_interface: bool,
    pub preferred_kinds: Vec<LinkKind>,
    pub discovery_multicast: SocketAddr,
    pub discovery_seeds: Vec<SocketAddr>,
    pub service_port: u16,
    pub advertise: bool,
}

impl Default for DataPlaneConfig {
    fn default() -> Self {
        Self {
            allowed_interfaces: BTreeSet::new(),
            brainstem_interfaces: ["wlan1".to_string()].into_iter().collect(),
            allow_brainstem_interface: false,
            preferred_kinds: vec![LinkKind::Ethernet, LinkKind::Wifi, LinkKind::Other],
            discovery_multicast: "239.255.78.87:8788".parse().unwrap(),
            discovery_seeds: Vec::new(),
            service_port: 22,
            advertise: true,
        }
    }
}

impl DataPlaneConfig {
    pub fn load(path: &Path) -> Result<Self> {
        Ok(toml::from_str(&fs::read_to_string(path).with_context(
            || format!("read data-plane config {}", path.display()),
        )?)?)
    }

    pub fn select_interfaces(
        &self,
        candidates: &[InterfaceAddress],
    ) -> Result<Vec<InterfaceAddress>> {
        let mut selected = candidates
            .iter()
            .filter(|candidate| candidate.up && !candidate.address.is_loopback())
            .filter(|candidate| {
                self.allowed_interfaces.is_empty()
                    || self.allowed_interfaces.contains(&candidate.name)
            })
            .filter(|candidate| {
                self.allow_brainstem_interface
                    || !self.brainstem_interfaces.contains(&candidate.name)
            })
            .cloned()
            .collect::<Vec<_>>();
        selected.sort_by_key(|candidate| {
            let kind_rank = self
                .preferred_kinds
                .iter()
                .position(|kind| kind == &candidate.kind)
                .unwrap_or(usize::MAX);
            (
                kind_rank,
                candidate.route_metric,
                Reverse(candidate.address.is_ipv4()),
            )
        });
        if selected.is_empty() {
            anyhow::bail!(
                "no eligible higher-brain interface; brainstem interfaces remain excluded"
            );
        }
        Ok(selected)
    }

    pub fn endpoints(&self, candidates: &[InterfaceAddress]) -> Result<Vec<ServiceEndpoint>> {
        Ok(self
            .select_interfaces(candidates)?
            .into_iter()
            .map(|interface| ServiceEndpoint {
                interface: interface.name,
                address: SocketAddr::new(interface.address, self.service_port),
                kind: interface.kind,
            })
            .collect())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServiceEndpoint {
    pub interface: String,
    pub address: SocketAddr,
    pub kind: LinkKind,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DiscoveryAdvertisement {
    pub schema_version: u32,
    pub role: String,
    pub capabilities: AcceleratorCapabilities,
    pub endpoints: Vec<ServiceEndpoint>,
}

/// Try advertised endpoints in their preference order. Callers provide the
/// transport operation (SSH, SFTP, rsync, or a future protocol), so one failed
/// link does not pin the session to a dead address.
pub fn try_endpoints<T, E>(
    endpoints: &[ServiceEndpoint],
    mut operation: impl FnMut(&ServiceEndpoint) -> std::result::Result<T, E>,
) -> std::result::Result<T, Vec<E>> {
    let mut errors = Vec::new();
    for endpoint in endpoints {
        match operation(endpoint) {
            Ok(value) => return Ok(value),
            Err(error) => errors.push(error),
        }
    }
    Err(errors)
}

pub fn advertise_once(
    config: &DataPlaneConfig,
    interfaces: &[InterfaceAddress],
    advertisement: &DiscoveryAdvertisement,
) -> Result<usize> {
    if !config.advertise {
        return Ok(0);
    }
    let payload = serde_json::to_vec(advertisement)?;
    let mut sent = 0;
    for interface in config.select_interfaces(interfaces)? {
        let socket = UdpSocket::bind(SocketAddr::new(interface.address, 0))?;
        if let (IpAddr::V4(local), IpAddr::V4(_)) =
            (interface.address, config.discovery_multicast.ip())
        {
            socket2::SockRef::from(&socket).set_multicast_if_v4(&local)?;
            if socket.send_to(&payload, config.discovery_multicast).is_ok() {
                sent += 1;
            }
        }
        for seed in &config.discovery_seeds {
            if socket.send_to(&payload, seed).is_ok() {
                sent += 1;
            }
        }
    }
    Ok(sent)
}

/// Enumerate addresses without assuming interface names, adapters, or a router.
/// `ip` is part of the provisioned `iproute2` package.
pub fn local_interfaces() -> Result<Vec<InterfaceAddress>> {
    let output = Command::new("ip")
        .args(["-j", "address", "show"])
        .output()
        .context("run ip -j address show")?;
    if !output.status.success() {
        anyhow::bail!("ip address enumeration failed");
    }
    let records: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    let route_metrics = local_route_metrics();
    let mut out = Vec::new();
    for record in records.as_array().into_iter().flatten() {
        let name = record["ifname"].as_str().unwrap_or_default().to_string();
        let up = record["operstate"].as_str() == Some("UP")
            || record["flags"]
                .as_array()
                .is_some_and(|flags| flags.iter().any(|flag| flag.as_str() == Some("UP")));
        let kind = if Path::new("/sys/class/net")
            .join(&name)
            .join("wireless")
            .exists()
        {
            LinkKind::Wifi
        } else if name.starts_with("en") || name.starts_with("eth") {
            LinkKind::Ethernet
        } else {
            LinkKind::Other
        };
        for info in record["addr_info"].as_array().into_iter().flatten() {
            if info["scope"].as_str() != Some("global") {
                continue;
            }
            if let Some(address) = info["local"].as_str().and_then(|value| value.parse().ok()) {
                out.push(InterfaceAddress {
                    name: name.clone(),
                    address,
                    kind,
                    up,
                    route_metric: route_metrics.get(&name).copied().unwrap_or(100),
                });
            }
        }
    }
    Ok(out)
}

fn local_route_metrics() -> BTreeMap<String, u32> {
    let Ok(output) = Command::new("ip").args(["-j", "route", "show"]).output() else {
        return BTreeMap::new();
    };
    let Ok(records) = serde_json::from_slice::<serde_json::Value>(&output.stdout) else {
        return BTreeMap::new();
    };
    let mut metrics = BTreeMap::new();
    for route in records.as_array().into_iter().flatten() {
        let Some(device) = route["dev"].as_str() else {
            continue;
        };
        let metric = route["metric"].as_u64().unwrap_or(100) as u32;
        metrics
            .entry(device.to_string())
            .and_modify(|current: &mut u32| *current = (*current).min(metric))
            .or_insert(metric);
    }
    metrics
}

pub fn receive_one(
    bind: SocketAddr,
    multicast: Option<Ipv4Addr>,
    timeout: std::time::Duration,
) -> Result<DiscoveryAdvertisement> {
    let socket = UdpSocket::bind(bind)?;
    if let (Some(group), IpAddr::V4(local)) = (multicast, bind.ip()) {
        socket.join_multicast_v4(&group, &local)?;
    }
    socket.set_read_timeout(Some(timeout))?;
    let mut buffer = vec![0u8; 64 * 1024];
    let (read, _) = socket.recv_from(&mut buffer)?;
    Ok(serde_json::from_slice(&buffer[..read])?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn iface(name: &str, ip: &str, kind: LinkKind, metric: u32) -> InterfaceAddress {
        InterfaceAddress {
            name: name.into(),
            address: ip.parse().unwrap(),
            kind,
            up: true,
            route_metric: metric,
        }
    }

    #[test]
    fn ethernet_is_preferred_and_brainstem_is_refused() {
        let config = DataPlaneConfig::default();
        let selected = config
            .select_interfaces(&[
                iface("wlan1", "192.168.4.2", LinkKind::Wifi, 1),
                iface("wlp2s0", "192.168.1.9", LinkKind::Wifi, 10),
                iface("enp3s0", "10.42.0.2", LinkKind::Ethernet, 50),
            ])
            .unwrap();
        assert_eq!(selected[0].name, "enp3s0");
        assert!(selected.iter().all(|item| item.name != "wlan1"));
    }

    #[test]
    fn explicit_allow_is_required_for_brainstem_interface() {
        let mut config = DataPlaneConfig::default();
        let only = [iface("wlan1", "192.168.4.2", LinkKind::Wifi, 1)];
        assert!(config.select_interfaces(&only).is_err());
        config.allow_brainstem_interface = true;
        assert_eq!(config.select_interfaces(&only).unwrap().len(), 1);
    }

    #[test]
    fn down_links_are_skipped_for_fallback() {
        let config = DataPlaneConfig::default();
        let mut ethernet = iface("eth9", "10.0.0.2", LinkKind::Ethernet, 1);
        ethernet.up = false;
        let selected = config
            .select_interfaces(&[ethernet, iface("wifi9", "10.1.0.2", LinkKind::Wifi, 20)])
            .unwrap();
        assert_eq!(selected[0].name, "wifi9");
    }

    #[test]
    fn failed_preferred_endpoint_falls_back_to_next_network() {
        let endpoints = vec![
            ServiceEndpoint {
                interface: "eth0".into(),
                address: "10.42.0.2:8787".parse().unwrap(),
                kind: LinkKind::Ethernet,
            },
            ServiceEndpoint {
                interface: "wifi0".into(),
                address: "192.168.1.2:8787".parse().unwrap(),
                kind: LinkKind::Wifi,
            },
        ];
        let selected = try_endpoints(&endpoints, |endpoint| {
            if endpoint.kind == LinkKind::Ethernet {
                Err("cable unplugged")
            } else {
                Ok(endpoint.interface.clone())
            }
        })
        .unwrap();
        assert_eq!(selected, "wifi0");
    }
}
