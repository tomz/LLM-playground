# distgpt/scripts/host_setup

Host-environment fixes needed before some distgpt examples can run.
These are NOT part of distgpt's runtime -- they're shell scripts that
adjust the *host* (drivers, modprobe configs, grub cmdline). They live
here because the 2-GPU FSDP smoke (`scripts/smoke_2gpu.sh`) won't have
two GPUs to use on i78700 without them.

## `i78700_enable_p100.sh`

Swaps the i78700 dev box from the NVIDIA **open** kernel module to the
**proprietary** one, and unbinds the Tesla P100 from `vfio-pci` (leftover
VM-passthrough config). Required because the open module refuses to
attach to Pascal GPUs (no GSP firmware on the silicon). The proprietary
driver handles Pascal + Blackwell side-by-side in one module, so the
RTX 5060 Ti keeps working.

Run once, then reboot. After reboot `nvidia-smi -L` shows both GPUs and
`scripts/smoke_2gpu.sh` can do a real 2-GPU FSDP run.

## `i78700_revert_to_open_driver.sh`

The rollback. Restores vfio capture of the P100 and reinstalls the
`-open` packages. Reboot to apply.

## `unbind-broken-nvidia.sh` (+ `.service`)

Runtime workaround used by the [`p100_416m_fineweb.md`](../../examples/p100_416m_fineweb.md)
example. After running `i78700_enable_p100.sh`, the proprietary
driver attaches the P100 cleanly but **fails `RmInitAdapter` on the
Blackwell 5060 Ti** (the proprietary 580 module has no GSP-loader for
Blackwell — that needs the `-open` variant, which in turn drops Pascal).
The half-broken state breaks `cuInit` for *every* CUDA process,
including ones that only want the P100, because `cuInit` enumerates
all GPUs the driver claims. Unbinding the broken slot at the PCI level
gets `cuInit` working again.

```bash
sudo scripts/host_setup/unbind-broken-nvidia.sh 0000:05:00.0
```

Or install the systemd unit to do it automatically at boot.

## Why these are committed to the repo

The distgpt README claims the framework runs end-to-end on real hardware.
For the single-GPU 5060 Ti example that's already true. For the 2-GPU
FSDP example we want to publish next, the host-driver state on the
actual machine we tested on is part of the reproduction recipe -- not
including it would make `smoke_2gpu.sh` mysteriously hang at NCCL init
for anyone with a mixed-vendor or mixed-generation NVIDIA setup.
