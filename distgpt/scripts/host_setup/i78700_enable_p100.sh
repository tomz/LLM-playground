#!/bin/bash
# Swap i78700 from nvidia-open to nvidia (proprietary) so the Tesla P100
# (Pascal, no GSP firmware) can be driven alongside the RTX 5060 Ti
# (Blackwell). Run on the host, NOT inside distgpt -- this is a
# host-environment fix, kept under distgpt/scripts/ because it's needed
# to make the 2-GPU FSDP smoke run actually run.
#
# Why the swap is needed
# ----------------------
# Ubuntu 26.04 ships the NVIDIA "open" kernel module by default. The open
# module requires GSP firmware on the GPU; Pascal (P100, GTX 1070) doesn't
# have it, so the open module refuses to attach:
#
#   NVRM: The NVIDIA GPU 0000:01:00.0 (PCI ID: 10de:15f8) installed in
#         this system is not supported by open nvidia.ko because it does
#         not include the required GPU System Processor (GSP).
#
# The *proprietary* driver supports Pascal AND Blackwell in the same
# kernel module, so swapping fixes the P100 without breaking the 5060 Ti.
#
# Both the -open and proprietary DKMS source trees were already installed
# on this box -- only the -open one had been built. So this script is
# mostly an apt remove + apt install + dkms rebuild, no big downloads.
#
# Plus: the P100 was previously bound to vfio-pci (leftover from a
# VM-passthrough setup) by /etc/modprobe.d/vfio.conf and a
# vfio-pci.ids=10de:15f8 entry on the kernel cmdline. We undo both so
# the proprietary nvidia driver claims the P100 at next boot.
#
# AFTER this script: REBOOT. Then `nvidia-smi -L` should show both GPUs.
# Rollback: scripts/host_setup/i78700_revert_to_open_driver.sh

set -euxo pipefail

# --- 1. Disable vfio capture of the P100 -------------------------------------
sudo mv /etc/modprobe.d/vfio.conf         /etc/modprobe.d/vfio.conf.disabled
sudo mv /etc/modprobe.d/vfio-softdep.conf /etc/modprobe.d/vfio-softdep.conf.disabled

# Strip vfio-pci.ids=... from the kernel command line
sudo sed -i 's| vfio-pci.ids=10de:15f8||' /etc/default/grub
sudo update-grub
sudo update-initramfs -u

# --- 2. Swap drivers: remove -open, install proprietary ----------------------
# The -open packages currently provide the active /lib/modules/.../nvidia.ko.
# Removing them and installing the proprietary meta-package triggers a DKMS
# rebuild from the already-installed nvidia-dkms-580 source tree.
sudo apt-get remove --purge -y \
    nvidia-driver-580-open \
    nvidia-dkms-580-open \
    nvidia-kernel-source-580-open

sudo apt-get install -y nvidia-driver-580

# --- 3. Verify the proprietary module is what got built ----------------------
sudo dkms status
modinfo nvidia | grep -E '^(filename|version|license)'
# Expect: license: NVIDIA  (not 'Dual MIT/GPL' as the open module reports)

cat <<'EOF'

==============================================================
DONE. Reboot now:    sudo reboot
After reboot, verify: nvidia-smi -L
  (expect TWO GPUs: 5060 Ti @ 05:00.0 + Tesla P100 @ 01:00.0)

Rollback script: distgpt/scripts/host_setup/i78700_revert_to_open_driver.sh
==============================================================
EOF
