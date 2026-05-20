"""Simulated pretraining loop. Iterates in chunks of "virtual" steps,
advances the clock, samples loss from the scaling law, samples failures,
posts events."""
from __future__ import annotations
import random
from dataclasses import dataclass

from .clock import Clock
from .cluster import Cluster, GPU_SPECS
from .economy import CostBook
from .events import EventBus
from .scaling import step_loss_curve, compute_flops


@dataclass
class PretrainSpec:
    n_params: float
    total_tokens: float
    seq_len: int
    global_batch_tokens: int
    target_mfu: float = 0.50
    spike_prob_per_1k_steps: float = 0.02
    log_every: int = 100
    # If set, use this measured seconds/step directly (skips the
    # peak_tflops * target_mfu calculation). Lets a real-GPU probe
    # calibrate the simulator's wall-clock to actual local hardware.
    measured_seconds_per_step: float | None = None
    measured_source: str | None = None  # free-form label for the event log


def simulate_pretrain(
    spec: PretrainSpec,
    cluster: Cluster,
    clock: Clock,
    cost: CostBook,
    bus: EventBus,
    seed: int = 0,
) -> dict:
    rng = random.Random(seed)
    total_steps = max(1, int(spec.total_tokens // spec.global_batch_tokens))
    # throughput
    achieved_tflops = cluster.peak_tflops * spec.target_mfu
    flops_per_step = compute_flops(spec.n_params, spec.global_batch_tokens)
    modeled_seconds_per_step = flops_per_step / (achieved_tflops * 1e12)
    if spec.measured_seconds_per_step is not None:
        seconds_per_step = spec.measured_seconds_per_step
        throughput_source = spec.measured_source or "measured"
    else:
        seconds_per_step = modeled_seconds_per_step
        throughput_source = "modeled"
    bus.emit("pretrain.start",
             n_params=spec.n_params, total_tokens=spec.total_tokens,
             total_steps=total_steps, seconds_per_step=seconds_per_step,
             modeled_seconds_per_step=modeled_seconds_per_step,
             throughput_source=throughput_source,
             cluster_gpus=cluster.total_gpus, mfu=spec.target_mfu,
             estimated_hours=seconds_per_step * total_steps / 3600)

    losses = []
    spikes = 0
    last_log = -spec.log_every
    chunk = max(1, spec.log_every)
    step = 0
    while step < total_steps:
        n = min(chunk, total_steps - step)
        dt = n * seconds_per_step
        # sample failures over this chunk (mutates cluster.downtime_seconds)
        cluster.tick(dt, rng)
        # downtime is added on top of training time
        dt_total = dt + (cluster.downtime_seconds - cost.by_resource.get("_dt_acc", 0.0))
        cost.by_resource["_dt_acc"] = cluster.downtime_seconds
        clock.advance(dt_total)
        # cost: GPU-hours for ALL GPUs even during downtime (we pay for the cluster)
        gpu_hours = cluster.total_gpus * (dt_total / 3600.0)
        dollars = gpu_hours * GPU_SPECS[cluster.gpu_type]["price"]
        cost.charge("pretrain", f"gpu_{cluster.gpu_type}", dollars)

        step += n
        loss = step_loss_curve(spec.n_params, spec.total_tokens, step - 1, total_steps)
        # spike sampling
        if rng.random() < spec.spike_prob_per_1k_steps * (n / 1000.0):
            spikes += 1
            loss *= 1.6  # ugly spike
            bus.emit("pretrain.spike", step=step, loss=loss)
        if step - last_log >= spec.log_every:
            losses.append({"step": step, "loss": loss, "day": clock.days})
            bus.emit("pretrain.log", step=step, loss=loss, day=clock.days,
                     failures=cluster.failures, healthy_gpus=cluster.healthy_gpus,
                     dollars=dollars)
            last_log = step

    bus.emit("pretrain.done",
             final_loss=losses[-1]["loss"], steps=total_steps, spikes=spikes,
             failures=cluster.failures, dollars=cost.by_phase.get("pretrain", 0))
    return {
        "total_steps": total_steps,
        "losses": losses,
        "final_loss": losses[-1]["loss"] if losses else None,
        "spikes": spikes,
        "failures": cluster.failures,
    }
