#!/bin/bash
# Undo i78700_enable_p100.sh. Restores the open-driver-only state the
# box had before the swap (5060 Ti works, P100 bound to vfio-pci).

set -euxo pipefail

# Restore vfio configs.
sudo mv /etc/modprobe.d/vfio.conf.disabled         /etc/modprobe.d/vfio.conf         || true
sudo mv /etc/modprobe.d/vfio-softdep.conf.disabled /etc/modprobe.d/vfio-softdep.conf || true

if ! grep -q 'vfio-pci.ids=10de:15f8' /etc/default/grub; then
    sudo sed -i 's|\(GRUB_CMDLINE_LINUX_DEFAULT="[^"]*\)"|\1 vfio-pci.ids=10de:15f8"|' /etc/default/grub
fi
sudo update-grub
sudo update-initramfs -u

sudo apt-get remove --purge -y nvidia-driver-580 nvidia-dkms-580 nvidia-kernel-source-580 || true
sudo apt-get install -y \
    nvidia-driver-580-open \
    nvidia-dkms-580-open \
    nvidia-kernel-source-580-open

sudo dkms status
modinfo nvidia | grep -E '^(filename|version|license)'
echo 'Reboot to apply.'
