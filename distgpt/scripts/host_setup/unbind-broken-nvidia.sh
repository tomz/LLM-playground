#!/bin/bash
# Unbind a non-initialising NVIDIA GPU from the nvidia driver at runtime.
#
# Why this exists
# ---------------
# On i78700 the proprietary 580 driver loads cleanly, attaches the Tesla
# P100 (Pascal, sm_60) -- but the RTX 5060 Ti (Blackwell, sm_120) fails
# RmInitAdapter with 0x22:0x56:897 (Blackwell needs the -open kernel
# modules; the proprietary one doesn't have GSP-loader support for it).
#
# That alone would be tolerable -- nvidia-smi hides the broken device --
# but `cuInit` (the CUDA driver-API call torch makes on first GPU use)
# enumerates EVERY GPU the nvidia driver claims, including the one stuck
# in RmInitAdapter failure, and returns CUDA_ERROR_INVALID_DEVICE (101)
# for the whole process. `CUDA_VISIBLE_DEVICES` does not help -- that
# filter runs *after* cuInit.
#
# The fix is to unbind the broken GPU at the PCI level so the nvidia
# driver releases it and `cuInit` only sees working devices.
#
# Usage: sudo ./unbind-broken-nvidia.sh 0000:05:00.0
#        sudo ./unbind-broken-nvidia.sh                 # auto-detects from dmesg
#
# Doesn't survive reboot. Install the systemd unit alongside this script
# (unbind-broken-nvidia.service) if you want it run automatically at boot.
set -euo pipefail

SLOT="${1:-}"
if [[ -z "$SLOT" ]]; then
  # Auto-detect: pick the most recent RmInitAdapter failure from dmesg.
  SLOT=$(dmesg | grep -oE 'GPU [0-9a-f:.]+: RmInitAdapter failed' | tail -1 \
          | awk '{print $2}' | tr -d ':')
  if [[ -z "$SLOT" ]]; then
    echo "usage: $0 <pci-slot, e.g. 0000:05:00.0>" >&2
    echo "  (no RmInitAdapter failure found in dmesg to auto-detect)" >&2
    exit 2
  fi
  echo "[auto] detected failing slot from dmesg: $SLOT"
fi

if [[ ! -e "/sys/bus/pci/devices/$SLOT" ]]; then
  echo "no PCI device at $SLOT" >&2
  exit 3
fi

if [[ ! -e "/sys/bus/pci/drivers/nvidia/$SLOT" ]]; then
  echo "$SLOT is not currently bound to nvidia (nothing to do)"
  exit 0
fi

echo "$SLOT" > /sys/bus/pci/drivers/nvidia/unbind
echo "unbound $SLOT from nvidia driver"

# Quick verification: cuInit should now succeed.
if command -v python3 >/dev/null; then
  python3 - <<'PY'
import ctypes
libcuda = ctypes.CDLL('libcuda.so.1')
rc = libcuda.cuInit(0)
cnt = ctypes.c_int(-1); libcuda.cuDeviceGetCount(ctypes.byref(cnt))
print(f"cuInit returned {rc}, cuDeviceGetCount = {cnt.value}")
PY
fi
