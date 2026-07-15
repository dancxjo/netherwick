#!/usr/bin/env bash
set -euo pipefail

umount_mode=0
if [ "${1:-}" = "--umount" ]; then
    umount_mode=1
    shift
fi

kernel_name="${1:-}"
if [ -z "$kernel_name" ]; then
    echo "usage: pico-bootsel-mount [--umount] KERNEL_NAME" >&2
    exit 2
fi

device="/dev/$kernel_name"
mount_base="${PICO_BOOTSEL_MOUNT_BASE:-/media}"
mount_user="${PICO_BOOTSEL_USER:-${SUDO_USER:-${USER:-}}}"
mount_group="${PICO_BOOTSEL_GROUP:-$mount_user}"

if [ -n "$mount_user" ] && id -u "$mount_user" >/dev/null 2>&1; then
    uid="$(id -u "$mount_user")"
else
    uid=0
fi

if [ -n "$mount_group" ] && getent group "$mount_group" >/dev/null 2>&1; then
    gid="$(getent group "$mount_group" | cut -d: -f3)"
else
    gid="$uid"
fi

if [ "$uid" = "0" ]; then
    mount_dir="${PICO_BOOTSEL_MOUNT_DIR:-$mount_base/RPI-RP2}"
else
    mount_dir="${PICO_BOOTSEL_MOUNT_DIR:-$mount_base/$mount_user/RPI-RP2}"
fi

if [ "$umount_mode" = "1" ]; then
    if mountpoint -q "$mount_dir"; then
        umount "$mount_dir"
    fi
    rmdir "$mount_dir" 2>/dev/null || true
    exit 0
fi

if [ ! -b "$device" ]; then
    echo "BOOTSEL block device not found: $device" >&2
    exit 1
fi

mkdir -p "$mount_dir"
if ! mountpoint -q "$mount_dir"; then
    mount -t vfat -o "uid=$uid,gid=$gid,umask=022,noatime,flush" "$device" "$mount_dir"
fi
chmod 755 "$mount_dir"
echo "Mounted RPI-RP2 at $mount_dir for uid=$uid gid=$gid"
